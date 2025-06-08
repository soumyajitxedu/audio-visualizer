import pyglet
from pyglet.gl import *
from pyglet.window import key, mouse
import librosa
import numpy as np
import random
import math
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
WINDOW_WIDTH = 1600 # Increased for more detail
WINDOW_HEIGHT = 900
NUM_NODES = 60      # Fine-tune for performance vs. density
NODE_DISTRIBUTION_RADIUS = 3.0
NODE_CORE_BASE_SIZE = 0.015
NODE_AURA_BASE_SCALE = 1.8 # Aura radius = core_radius * scale
NODE_SIZE_MULTIPLIER = 0.10
CONNECTION_THRESHOLD_DISTANCE = 2.0
FPS = 60.0
LABEL_OFFSET_Y = 0.03
LABEL_FONT_SIZE = 7

NUM_STARFIELD_PARTICLES = 300
STARFIELD_RADIUS = 15.0

# --- Global Variables for Pyglet part ---
pyglet_app_running = False
audio_data = None
sr = None
hop_length = 512 # For STFT, RMS, Centroid. Affects responsiveness.

# Camera control
camera_rot_x = 15
camera_rot_y = -10
camera_zoom = -7.0

# Data structures
nodes_list = []
connections = []
pyglet_batch = pyglet.graphics.Batch()
connection_batch = pyglet.graphics.Batch()
connection_vertex_lists = []
starfield_batch = pyglet.graphics.Batch()
star_particles = [] # Store star particle vertex lists for animation

# --- Helper Functions ---
def hsv_to_rgb(h, s, v):
    # H in [0,360], S,V in [0,1]
    if s == 0.0: return (v, v, v)
    i = math.floor(h / 60.0)
    f = (h / 60.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)
    return (0,0,0)

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

# --- Star Particle Class (Simple) ---
class StarParticle:
    def __init__(self, batch):
        self.pos = np.array([
            random.uniform(-STARFIELD_RADIUS, STARFIELD_RADIUS),
            random.uniform(-STARFIELD_RADIUS, STARFIELD_RADIUS),
            random.uniform(-STARFIELD_RADIUS, STARFIELD_RADIUS)
        ])
        # Make distant stars smaller/dimmer
        dist_sq = self.pos[0]**2 + self.pos[1]**2 + self.pos[2]**2
        brightness_factor = clamp(1.0 - (dist_sq / (STARFIELD_RADIUS**2 * 1.5)), 0.1, 0.8)
        
        self.color = (int(200 * brightness_factor), int(200 * brightness_factor), int(220 * brightness_factor), int(150 * brightness_factor))
        self.size = random.uniform(1, 3) * brightness_factor
        
        # Using pyglet.shapes.Circle for pseudo-3D point sprites (always face camera)
        # For true points: batch.add(1, GL_POINTS, None, ('v3f', self.pos), ('c4B', self.color)) and glPointSize
        self.shape = pyglet.shapes.Circle(self.pos[0], self.pos[1], self.size, color=self.color[:3], batch=batch)
        self.shape.opacity = self.color[3]
        # Note: pyglet.shapes.Circle is 2D. For 3D points that scale with distance, you'd use GL_POINTS.
        # For this aesthetic, we'll draw them as fixed-size points in 3D space.
        # So, we will switch to GL_POINTS for the starfield.
        self.vertex_list = starfield_batch.add(1, GL_POINTS, None, ('v3f/stream', self.pos), ('c4B/static', self.color))
        self.z_velocity = random.uniform(-0.001, 0.001) # Slow drift

    def update(self, dt):
        self.pos[2] += self.z_velocity # Slow drift
        if self.pos[2] > STARFIELD_RADIUS: self.pos[2] = -STARFIELD_RADIUS
        if self.pos[2] < -STARFIELD_RADIUS: self.pos[2] = STARFIELD_RADIUS
        self.vertex_list.vertices = self.pos

# --- Node Class ---
class Node:
    def __init__(self, id_num, x, y, z, batch):
        self.id = id_num
        self.pos = np.array([x, y, z], dtype=float)
        
        self.core_base_size = NODE_CORE_BASE_SIZE
        self.current_core_size = self.core_base_size
        self.target_core_size = self.core_base_size
        
        self.aura_scale = NODE_AURA_BASE_SCALE
        self.current_aura_size = self.current_core_size * self.aura_scale
        self.target_aura_size = self.target_core_size * self.aura_scale

        self.activation_level = 0.0 # Normalized (0-1)

        self.core_color_rgb = (0.8, 0.8, 0.8)
        self.aura_color_rgb = (0.5, 0.5, 0.7)
        self.target_core_hsv = (240.0, 0.7, 0.6) # Hue, Sat, Val
        self.target_aura_hsv = (240.0, 0.5, 0.4)

        self.core_opacity = 220 # 0-255
        self.aura_opacity = 0   # 0-255, starts invisible

        self.sphere_core = pyglet.shapes.Sphere(
            radius=self.current_core_size, subdivisions=2, # Lower subdivisions for performance
            color=(int(c*255) for c in self.core_color_rgb), batch=batch)
        self.sphere_core.opacity = self.core_opacity
        
        self.sphere_aura = pyglet.shapes.Sphere(
            radius=self.current_aura_size, subdivisions=2,
            color=(int(c*255) for c in self.aura_color_rgb), batch=batch)
        self.sphere_aura.opacity = self.aura_opacity

        self.label = pyglet.text.Label(
            f"{self.activation_level:.2f}", font_name='Arial', font_size=LABEL_FONT_SIZE,
            x=self.pos[0], y=self.pos[1] + LABEL_OFFSET_Y, z=self.pos[2],
            anchor_x='center', anchor_y='bottom', batch=batch, color=(180, 180, 200, 180))
        
        self.set_positions()

    def set_positions(self):
        self.sphere_core.x, self.sphere_core.y, self.sphere_core.z = self.pos
        self.sphere_aura.x, self.sphere_aura.y, self.sphere_aura.z = self.pos
        self.label.x, self.label.y, self.label.z = self.pos[0], self.pos[1] + LABEL_OFFSET_Y + self.current_core_size, self.pos[2]

    def update_visuals(self, intensity, bass_level, spectral_centroid_norm):
        self.activation_level = intensity
        self.target_core_size = self.core_base_size + intensity * NODE_SIZE_MULTIPLIER
        self.target_aura_size = self.target_core_size * (self.aura_scale + intensity * 0.5) # Aura grows more with intensity

        # Core Color
        # Hue: Bass (low bass ~240-300 (blue/purple), high bass ~0-60 (red/orange))
        core_hue = clamp(270 - (bass_level * 270), 0, 330) 
        core_sat = clamp(0.6 + intensity * 0.4, 0.5, 1.0)
        core_val = clamp(0.5 + intensity * 0.5, 0.4, 1.0)
        self.target_core_hsv = (core_hue, core_sat, core_val)

        # Aura Color (subtler, perhaps complementary or shifted hue)
        # Spectral centroid influences aura hue or brightness. Higher centroid = "sharper" color (e.g. toward cyan/yellow)
        aura_hue_shift = (spectral_centroid_norm - 0.5) * 60 # Shift hue by up to +/- 30 degrees
        aura_hue = (core_hue + aura_hue_shift + 360) % 360
        aura_sat = clamp(0.4 + intensity * 0.3 + spectral_centroid_norm * 0.2, 0.3, 0.8)
        aura_val = clamp(0.3 + intensity * 0.4 + spectral_centroid_norm * 0.3, 0.2, 0.7)
        self.target_aura_hsv = (aura_hue, aura_sat, aura_val)
        
        self.aura_opacity_target = int(clamp(intensity * 150 + spectral_centroid_norm * 50, 20, 180))


    def smooth_update(self, dt):
        lerp_factor = 0.12
        self.current_core_size += (self.target_core_size - self.current_core_size) * lerp_factor
        self.current_aura_size += (self.target_aura_size - self.current_aura_size) * lerp_factor
        
        # Smooth Core Color (HSV lerp is complex for hue, so direct RGB lerp after conversion)
        target_core_rgb = hsv_to_rgb(*self.target_core_hsv)
        self.core_color_rgb = tuple(self.core_color_rgb[i] + (target_core_rgb[i] - self.core_color_rgb[i]) * lerp_factor for i in range(3))

        # Smooth Aura Color
        target_aura_rgb = hsv_to_rgb(*self.target_aura_hsv)
        self.aura_color_rgb = tuple(self.aura_color_rgb[i] + (target_aura_rgb[i] - self.aura_color_rgb[i]) * lerp_factor for i in range(3))
        
        self.aura_opacity += (self.aura_opacity_target - self.aura_opacity) * lerp_factor

        # Update Pyglet shapes
        self.sphere_core.radius = self.current_core_size
        self.sphere_core.color = tuple(int(c*255) for c in self.core_color_rgb)
        self.sphere_core.opacity = self.core_opacity
        
        self.sphere_aura.radius = self.current_aura_size
        self.sphere_aura.color = tuple(int(c*255) for c in self.aura_color_rgb)
        self.sphere_aura.opacity = int(self.aura_opacity)

        self.label.text = f"{self.activation_level:.2f}"
        label_brightness = int(clamp(180 + self.activation_level * 75, 150, 255))
        self.label.color = (label_brightness, label_brightness, label_brightness, 200)
        
        self.set_positions()


# --- Pyglet Window Class ---
class VisualizerWindow(pyglet.window.Window):
    def __init__(self, audio_file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        glClearColor(0.0, 0.0, 0.0, 1.0) # Black background

        self.audio_file_path = audio_file_path
        self.player = pyglet.media.Player()
        self.time_elapsed = 0.0 # For shimmer effect

        self.load_audio_and_setup()
        if audio_data is None:
            self.close()
            return

        self.setup_gl()
        self.grid_batch = pyglet.graphics.Batch() # Keep the grid, but make it very subtle
        self.create_grid()
        self.create_starfield()

        pyglet.clock.schedule_interval(self.update, 1.0 / FPS)
        self.player.play()
        global pyglet_app_running
        pyglet_app_running = True

    def load_audio_and_setup(self):
        global audio_data, sr, nodes_list, connections, pyglet_batch, connection_batch, connection_vertex_lists, starfield_batch, star_particles
        try:
            audio_data, sr = librosa.load(self.audio_file_path, sr=None)
            source = pyglet.media.load(self.audio_file_path, streaming=False)
            self.player.queue(source)
        except Exception as e:
            print(f"Error loading audio: {e}")
            audio_data = None
            return

        nodes_list.clear()
        connections.clear()
        connection_vertex_lists.clear()
        star_particles.clear()

        pyglet_batch = pyglet.graphics.Batch() # For nodes and labels
        connection_batch = pyglet.graphics.Batch() # For connections
        starfield_batch = pyglet.graphics.Batch() # For stars

        for i in range(NUM_NODES):
            phi = random.uniform(0, 2 * math.pi)
            costheta = random.uniform(-1, 1)
            theta = math.acos(costheta)
            u = random.uniform(0, 1)**(1/3) # More uniform spherical dist
            r = NODE_DISTRIBUTION_RADIUS * u
            
            x = r * math.sin(theta) * math.cos(phi)
            y = r * math.sin(theta) * math.sin(phi)
            z = r * math.cos(theta)
            nodes_list.append(Node(i, x, y, z, pyglet_batch))

        for i in range(NUM_NODES):
            for j in range(i + 1, NUM_NODES):
                dist = np.linalg.norm(nodes_list[i].pos - nodes_list[j].pos)
                if dist < CONNECTION_THRESHOLD_DISTANCE:
                    connections.append((i,j))
                    vl = connection_batch.add(2, GL_LINES, None, ('v3f/stream', (0,0,0,0,0,0)), ('c4B/stream', (0,0,0,0)*2))
                    connection_vertex_lists.append(vl)
        print(f"Setup: {len(nodes_list)} nodes, {len(connections)} connections.")

    def create_starfield(self):
        global star_particles
        for _ in range(NUM_STARFIELD_PARTICLES):
            star_particles.append(StarParticle(starfield_batch))

    def create_grid(self): # Make grid more subtle
        grid_size = 10
        step = 2.0
        lines_data = []
        z_offset = -NODE_DISTRIBUTION_RADIUS * 1.5 
        for i_coord in np.arange(-grid_size, grid_size + step, step):
            lines_data.extend([-grid_size, i_coord, z_offset, grid_size, i_coord, z_offset])
            lines_data.extend([i_coord, -grid_size, z_offset, i_coord, grid_size, z_offset])
        num_points = len(lines_data) // 3
        color_data = (15, 15, 20) * num_points # Very dark blue/gray
        self.grid_batch.add(num_points, GL_LINES, None, ('v3f/static', lines_data), ('c3B/static', color_data))

    def setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # Standard alpha blending
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive blending for brighter glows (optional)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glPointSize(2.0) # For starfield if using GL_POINTS

    def on_resize(self, width, height):
        if height == 0: height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(55.0, width / float(height), 0.1, 100.0) # Adjusted FOV
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        global camera_rot_x, camera_rot_y
        if buttons & mouse.LEFT:
            camera_rot_x -= dy * 0.3 # Slower rotation
            camera_rot_y += dx * 0.3
    
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        global camera_zoom
        camera_zoom += scroll_y * 0.3
        camera_zoom = clamp(camera_zoom, -20, -2.5)

    def update(self, dt):
        self.time_elapsed += dt
        if audio_data is None or self.player.source is None or not self.player.playing:
            intensity, bass_level, centroid_norm = 0.0, 0.0, 0.0
            if self.player.source and self.player.time >= self.player.source.duration - 0.1 : # If near end
                # Optionally fade out player volume
                self.player.volume = max(0, self.player.volume - 0.05 * dt * 10) 
                if self.player.volume == 0: self.player.pause()
        else:
            current_audio_time = self.player.time
            start_sample = int(max(0, current_audio_time - hop_length / (2*sr)) * sr) # More precise windowing
            end_sample = int(min(len(audio_data)/sr, current_audio_time + hop_length / (2*sr)) * sr)
            
            frame_audio = audio_data[start_sample:end_sample]

            if len(frame_audio) < hop_length // 8:
                intensity, bass_level, centroid_norm = 0.0, 0.0, 0.0
            else:
                rms = librosa.feature.rms(y=frame_audio, frame_length=len(frame_audio), hop_length=len(frame_audio)+1)[0][0]
                intensity = clamp(rms / 0.30, 0, 1) # Adjusted normalization factor

                stft = librosa.stft(frame_audio, n_fft=hop_length*2, hop_length=len(frame_audio)+1)
                magnitude_spec = np.abs(stft)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=hop_length*2)
                
                bass_bin_max_freq = 200 # Hz
                bass_bins = freqs < bass_bin_max_freq
                bass_energy = np.sum(magnitude_spec[bass_bins]) if np.any(bass_bins) else 0
                
                mid_bins_min_freq = 200
                mid_bins_max_freq = 2000
                mid_bins = (freqs >= mid_bins_min_freq) & (freqs < mid_bins_max_freq)
                mid_energy = np.sum(magnitude_spec[mid_bins]) if np.any(mid_bins) else 0
                
                total_energy = np.sum(magnitude_spec) + 1e-6
                bass_level = clamp((bass_energy / total_energy) * 2.5, 0, 1) # Amplify bass proportion

                # Spectral Centroid
                centroid = librosa.feature.spectral_centroid(y=frame_audio, sr=sr, n_fft=hop_length*2, hop_length=len(frame_audio)+1)[0][0]
                centroid_norm = clamp(centroid / (sr / 4), 0, 1) # Normalize (sr/4 is a rough upper bound)

        for node in nodes_list:
            node.update_visuals(intensity, bass_level, centroid_norm)
            node.smooth_update(dt)
        
        self.update_connection_visuals(intensity, centroid_norm)

        for star in star_particles:
            star.update(dt)

    def update_connection_visuals(self, overall_intensity, spectral_centroid_norm):
        if not connection_vertex_lists or len(connection_vertex_lists) != len(connections): return

        for idx, (i, j) in enumerate(connections):
            node1, node2 = nodes_list[i], nodes_list[j]
            
            avg_activation = (node1.activation_level + node2.activation_level) / 2.0
            
            # Connection color: average of connected nodes' core colors, then desaturate/darken
            n1_core_hsv = node1.target_core_hsv
            n2_core_hsv = node2.target_core_hsv

            # Weighted average hue (careful with circular mean)
            # Simple average for now, can get complex with hue wrap-around
            avg_hue = (n1_core_hsv[0] + n2_core_hsv[0]) / 2.0 
            if abs(n1_core_hsv[0] - n2_core_hsv[0]) > 180: # Handle wrap-around
                 avg_hue = (avg_hue + 180) % 360

            avg_sat = (n1_core_hsv[1] + n2_core_hsv[1]) / 2.0 * 0.6 # Desaturate
            avg_val = (n1_core_hsv[2] + n2_core_hsv[2]) / 2.0 * 0.5 # Darken

            # Spectral centroid adds a "sparkle" or slight color shift to active connections
            highlight_shift = (spectral_centroid_norm - 0.5) * 30
            final_hue = (avg_hue + highlight_shift + 360) % 360
            final_sat = clamp(avg_sat + avg_activation * 0.3 + spectral_centroid_norm * 0.1, 0.2, 0.8)
            final_val = clamp(avg_val + avg_activation * 0.4 + spectral_centroid_norm * 0.2, 0.1, 0.7)

            conn_rgb = hsv_to_rgb(final_hue, final_sat, final_val)
            
            # Opacity based on average activation and overall intensity
            # Shimmer effect using time
            shimmer = (math.sin(self.time_elapsed * 5.0 + (i+j)*0.1) + 1) / 2.0 # 0 to 1
            alpha = int(clamp(avg_activation * 180 + overall_intensity * 50 + shimmer * avg_activation * 50, 10, 200))

            color_data = (int(conn_rgb[0]*255), int(conn_rgb[1]*255), int(conn_rgb[2]*255), alpha) * 2
            
            vl = connection_vertex_lists[idx]
            vl.vertices = [*node1.pos, *node2.pos]
            vl.colors = color_data
            
    def on_draw(self):
        self.clear()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glTranslatef(0, 0, camera_zoom)
        glRotatef(camera_rot_x, 1, 0, 0)
        glRotatef(camera_rot_y, 0, 1, 0)

        # Draw Starfield (drawn first, behind everything)
        glEnable(GL_POINT_SMOOTH) # For nicer points
        glPointSize(random.uniform(1.0,2.5)) # Dynamic point size for twinkling - apply per point if possible or globally
        starfield_batch.draw()
        glDisable(GL_POINT_SMOOTH)

        self.grid_batch.draw() # Subtle grid

        # Connections - line width based on average activation
        avg_node_activation = sum(n.activation_level for n in nodes_list) / len(nodes_list) if nodes_list else 0
        line_width = clamp(0.5 + avg_node_activation * 2.5, 0.5, 2.0)
        glLineWidth(line_width)
        connection_batch.draw()

        pyglet_batch.draw() # Nodes (spheres) and labels

    def on_close(self):
        global pyglet_app_running
        if self.player:
            self.player.pause()
            self.player.delete()
        pyglet_app_running = False
        super().on_close()

# --- Tkinter GUI (same as before) ---
def select_audio_file_and_run():
    root = tk.Tk()
    root.withdraw()
    audio_file = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=(("Audio Files", "*.mp3 *.wav *.ogg *.flac"), ("All files", "*.*"))
    )
    root.destroy()

    if audio_file:
        if pyglet.app.event_loop.is_running: pyglet.app.exit() # Ensure clean restart
        try:
            window = VisualizerWindow(audio_file_path=audio_file,
                                      width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
                                      caption='Aesthetic Audio Nebula', resizable=True, vsync=True) # vsync=True for smoother animation
            if pyglet_app_running:
                 pyglet.app.run()
        except Exception as e:
            print(f"Error setting up Pyglet: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No audio file selected.")

if __name__ == '__main__':
    select_audio_file_and_run()
    print("Visualizer program finished.")
