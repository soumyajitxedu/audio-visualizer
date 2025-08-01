import sys
import numpy as np
import sounddevice as sd
import time
import math
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QAction, QFileDialog, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QRadialGradient

import librosa

# --- Audio Processing Thread with Beat Detection ---
class AudioThread(QThread):
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, chunk_size=2048):
        super().__init__()
        self.chunk_size = chunk_size
        self.file_path = file_path
        self.running = True
        self.beat_times = []
        self.start_time = None
        
    def run(self):
        try:
            # Load audio and detect beats
            audio_data, self.sample_rate = librosa.load(self.file_path, sr=None, mono=False)
            if audio_data.ndim == 1: 
                audio_data = np.vstack([audio_data, audio_data])
            
            # Beat detection on mono mix
            mono_for_beats = librosa.to_mono(audio_data)
            tempo, beats = librosa.beat.beat_track(y=mono_for_beats, sr=self.sample_rate)
            self.beat_times = librosa.frames_to_time(beats, sr=self.sample_rate)
            
            left_channel, right_channel = audio_data[0], audio_data[1]
            mono_mix_float = librosa.to_mono(audio_data)
            
            sd.play(audio_data.T, self.sample_rate)
            num_chunks = len(mono_mix_float) // self.chunk_size
            self.start_time = time.time()
            
            for i in range(num_chunks):
                if not self.running: break
                
                expected_time = self.start_time + (i * self.chunk_size / self.sample_rate)
                while time.time() < expected_time: 
                    time.sleep(0.001)
                
                mono_chunk = mono_mix_float[i * self.chunk_size:(i + 1) * self.chunk_size]
                left_chunk = left_channel[i * self.chunk_size:(i + 1) * self.chunk_size]
                right_chunk = right_channel[i * self.chunk_size:(i + 1) * self.chunk_size]
                
                if mono_chunk.size == 0: continue
                
                fft_data = np.fft.rfft(mono_chunk * np.hanning(len(mono_chunk)))
                fft_freq = np.fft.rfftfreq(len(mono_chunk), 1.0 / self.sample_rate)
                magnitudes = np.abs(fft_data)
                
                mono_loudness = np.sqrt(np.mean(mono_chunk**2))
                left_loudness = np.sqrt(np.mean(left_chunk**2))
                right_loudness = np.sqrt(np.mean(right_chunk**2))
                
                # Current playback time for beat synchronization
                current_time = (i * self.chunk_size) / self.sample_rate
                
                self.data_ready.emit({
                    'magnitudes': magnitudes, 
                    'fft_freq': fft_freq,
                    'mono_loudness': mono_loudness, 
                    'left_loudness': left_loudness,
                    'right_loudness': right_loudness,
                    'beat_times': self.beat_times,
                    'current_time': current_time
                })
                
        except Exception as e:
            self.error_occurred.emit(f"Audio loading/processing error: {e}")
        finally:
            sd.stop()
    
    def stop(self):
        self.running = False
        sd.stop()

# --- Enhanced Star Class with States ---
class Star:
    NORMAL, MERGING, GLOWING = 0, 1, 2
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()
        self.state = self.NORMAL
        self.target_pos = None
        self.merge_speed_factor = 0.1
        
    def reset(self):
        self.x = random.uniform(-self.width / 2, self.width / 2)
        self.y = random.uniform(-self.height / 2, self.height / 2)
        self.z = random.uniform(1, self.width)
        self.state = self.NORMAL
        self.target_pos = None
        
    def update(self, speed, balance_shift=0):
        if self.state == self.NORMAL:
            self.z -= speed
            if self.z <= 0:
                self.reset()
                # Bias reset position based on audio balance
                if abs(balance_shift) > 0.1:
                    self.x += balance_shift * self.width * 0.2
        elif self.state == self.MERGING:
            if self.target_pos:
                # Calculate screen coordinates for merging logic
                factor = self.width / self.z
                current_screen_x = self.x * factor + self.width / 2
                current_screen_y = self.y * factor + self.height / 2

                dx = self.target_pos.x() - current_screen_x
                dy = self.target_pos.y() - current_screen_y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 20:  # Close enough to merge
                    self.state = self.GLOWING
                    return True  # Indicate it has merged
                    
                # Move towards target
                self.x += dx * self.merge_speed_factor * (self.z / self.width)
                self.y += dy * self.merge_speed_factor * (self.z / self.width)
                self.z -= speed * 1.5 # Move faster towards screen
                
                if self.z <= 1:
                    self.state = self.GLOWING
                    return True
                
        return False  # Not merged yet
    
    def get_screen_coords(self, balance_shift=0):
        if self.z < 1:
            return None, None, None
            
        factor = self.width / self.z
        x2d = self.x * factor + self.width / 2 + balance_shift
        y2d = self.y * factor + self.height / 2
        radius = (1 - self.z / self.width) * 3
        
        return x2d, y2d, radius
    
    def get_color(self, base_brightness, balance):
        if self.state == self.MERGING:
            return QColor(255, 255, 220, 255) # Bright white/yellow
        else:
            r = int(max(0, min(255, base_brightness - balance * 80)))
            b = int(max(0, min(255, base_brightness + balance * 80)))
            g = int(max(0, min(255, base_brightness - abs(balance) * 40)))
            return QColor(r, g, b, 200)

# --- Glow Effect Class for Collisions ---
class GlowEffect:
    def __init__(self, center_x, center_y, max_radius, duration_frames, color):
        self.center_x = center_x
        self.center_y = center_y
        self.max_radius = max_radius
        self.duration_frames = duration_frames
        self.current_frame = 0
        self.color = color
        self.burst_particles = []
        
    def update(self):
        self.current_frame += 1
        
        # Generate burst effects at peak
        if self.current_frame == self.duration_frames // 4 and not self.burst_particles:
            for _ in range(random.randint(15, 25)):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(2, 6)
                self.burst_particles.append({
                    'x': self.center_x,
                    'y': self.center_y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': random.randint(20, 30),
                    'max_life': 30,
                    'size': random.uniform(1.5, 3.5)
                })
        
        # Update particles
        for particle in self.burst_particles:
            if particle['life'] > 0:
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                particle['life'] -= 1
                particle['vx'] *= 0.98 # Air resistance
                particle['vy'] *= 0.98

        self.burst_particles = [p for p in self.burst_particles if p['life'] > 0]
        
        return self.current_frame < self.duration_frames
    
    def draw(self, painter):
        progress = self.current_frame / self.duration_frames
        
        # Main glow with expand-then-shrink radius
        eased_progress = 1 - (1 - progress) ** 3 # Ease-out cubic
        current_radius = self.max_radius * eased_progress
        alpha_mult = 1 - progress
        
        # Draw main glow
        if current_radius > 0:
            grad = QRadialGradient(self.center_x, self.center_y, current_radius)
            base_alpha = int(180 * alpha_mult)
            grad.setColorAt(0, QColor(self.color.red(), self.color.green(), self.color.blue(), base_alpha))
            grad.setColorAt(0.7, QColor(self.color.red(), self.color.green(), self.color.blue(), base_alpha // 3))
            grad.setColorAt(1, Qt.transparent)
            
            painter.setBrush(grad)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(self.center_x, self.center_y), current_radius, current_radius)
        
        # Draw burst particles
        painter.setPen(Qt.NoPen)
        for particle in self.burst_particles:
            particle_alpha = int(255 * (particle['life'] / particle['max_life']))
            painter.setBrush(QColor(255, 255, 220, particle_alpha))
            painter.drawEllipse(QPointF(particle['x'], particle['y']), particle['size'], particle['size'])

# --- Enhanced Visualizer Widget ---
class CircularVisualizerWidget(QWidget):
    NUM_BARS = 120
    MIN_FREQ = 20
    MAX_FREQ = 22000
    NUM_STARS = 400
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.magnitudes = np.zeros(self.NUM_BARS)
        self.mono_loudness = self.left_loudness = self.right_loudness = 0.0
        self.smoothed_magnitudes = np.zeros(self.NUM_BARS)
        self.smoothed_mono_loudness = self.smoothed_left_loudness = self.smoothed_right_loudness = 0.0
        self.log_freq_indices = self.create_log_freq_indices()
        self.stars = []
        
        # Enhanced features
        self.active_glows = []
        self.beat_times = []
        self.next_beat_idx = 0
        self.current_audio_time = 0.0
        self.screen_shake_intensity = 0.0
        self.flash_alpha = 0
        self.audio_balance = 0.0
        self.last_beat_time = -1.0
        self.beat_cooldown = 0.2
        self.background_pulse = 0.0
        self.beat_scale_factor = 1.0
        
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(1000 // 60)  # 60 FPS
        self.render_timer.timeout.connect(self._update_frame)
        self.render_timer.start()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.stars:
             self.stars = [Star(self.width(), self.height()) for _ in range(self.NUM_STARS)]
        for star in self.stars:
            star.width = self.width()
            star.height = self.height()
    
    def create_log_freq_indices(self):
        return np.logspace(np.log10(self.MIN_FREQ), np.log10(self.MAX_FREQ), self.NUM_BARS + 1)
    
    def update_visuals(self, audio_data):
        self.magnitudes = self.process_fft_data(audio_data['magnitudes'], audio_data['fft_freq'])
        self.mono_loudness = audio_data['mono_loudness']
        self.left_loudness = audio_data['left_loudness']
        self.right_loudness = audio_data['right_loudness']
        self.beat_times = audio_data.get('beat_times', self.beat_times)
        self.current_audio_time = audio_data.get('current_time', self.current_audio_time)
    
    def process_fft_data(self, raw_mags, raw_freqs):
        processed_mags = np.zeros(self.NUM_BARS)
        for i in range(self.NUM_BARS):
            min_f, max_f = self.log_freq_indices[i], self.log_freq_indices[i+1]
            indices = np.where((raw_freqs >= min_f) & (raw_freqs < max_f))
            if indices[0].size > 0: 
                processed_mags[i] = np.mean(raw_mags[indices])
        return processed_mags
    
    def _update_frame(self):
        # Smooth audio data
        self.smoothed_magnitudes = self.smoothed_magnitudes * 0.75 + self.magnitudes * 0.25
        self.smoothed_mono_loudness = self.smoothed_mono_loudness * 0.90 + self.mono_loudness * 0.10
        self.smoothed_left_loudness = self.smoothed_left_loudness * 0.85 + self.left_loudness * 0.15
        self.smoothed_right_loudness = self.smoothed_right_loudness * 0.85 + self.right_loudness * 0.15

        # Calculate audio balance
        epsilon = 1e-6
        total_loudness = self.smoothed_left_loudness + self.smoothed_right_loudness
        self.audio_balance = (self.smoothed_right_loudness - self.smoothed_left_loudness) / (total_loudness + epsilon) if total_loudness > epsilon else 0.0

        # Update stars
        star_speed = 0.5 + self.smoothed_mono_loudness * 25
        stars_to_reset = [star for star in self.stars if star.update(star_speed, self.audio_balance)]
        
        # Beat detection and collision effects
        time_since_last_beat = self.current_audio_time - self.last_beat_time
        if self.next_beat_idx < len(self.beat_times) and \
           self.current_audio_time >= self.beat_times[self.next_beat_idx] and \
           time_since_last_beat > self.beat_cooldown:
            self._trigger_collision_effect()
            self.last_beat_time = self.beat_times[self.next_beat_idx]
            self.next_beat_idx += 1
        
        # Update effects
        self.active_glows = [glow for glow in self.active_glows if glow.update()]
        self.background_pulse = self.background_pulse * 0.95 + self.smoothed_mono_loudness * 0.05
        self.beat_scale_factor = 1.0 + (self.beat_scale_factor - 1.0) * 0.9
        
        # Decay screen effects
        self.screen_shake_intensity *= 0.9
        self.flash_alpha = max(0, self.flash_alpha - 10)
        
        for star in stars_to_reset:
            star.reset()
        
        self.update()
    
    def _trigger_collision_effect(self):
        self.beat_scale_factor = 1.2 + self.smoothed_mono_loudness
        glow_center_x = self.width() / 2 - self.audio_balance * (self.width() * 0.2)
        glow_center_y = self.height() / 2 + random.uniform(-30, 30)
        
        glow_color = QColor(255, 200, 100)
        max_radius = min(self.width(), self.height()) * 0.2 * (1 + self.smoothed_mono_loudness * 2)
        self.active_glows.append(GlowEffect(glow_center_x, glow_center_y, max_radius, 40, glow_color))
        
        available_stars = [s for s in self.stars if s.state == Star.NORMAL]
        num_stars_to_merge = min(int(10 + self.smoothed_mono_loudness * 40), len(available_stars))
        
        if available_stars and num_stars_to_merge > 0:
            selected_stars = random.sample(available_stars, num_stars_to_merge)
            for star in selected_stars:
                star.state = Star.MERGING
                star.target_pos = QPointF(glow_center_x, glow_center_y)
                star.merge_speed_factor = random.uniform(0.08, 0.15)
        
        self.screen_shake_intensity = 4.0 + self.smoothed_mono_loudness * 8
        self.flash_alpha = min(150, int(80 + self.smoothed_mono_loudness * 150))
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.save()
        
        bg_intensity = int(self.background_pulse * 30)
        painter.fillRect(self.rect(), QColor(bg_intensity, bg_intensity, int(bg_intensity*1.2)))
        
        if self.screen_shake_intensity > 0.1:
            shake_x = random.uniform(-self.screen_shake_intensity, self.screen_shake_intensity)
            shake_y = random.uniform(-self.screen_shake_intensity, self.screen_shake_intensity)
            painter.translate(shake_x, shake_y)
        
        painter.setPen(Qt.NoPen)
        global_star_shift = -self.audio_balance * self.width() * 0.1
        
        for star in self.stars:
            if star.state == Star.MERGING: continue
            x, y, radius = star.get_screen_coords(global_star_shift)
            if x is not None and radius > 0.5:
                brightness = int(255 * (1 - star.z / star.width))
                color = star.get_color(brightness, self.audio_balance)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(x, y), radius, radius)
        
        painter.fillRect(self.rect(), QColor(0, 0, 0, 40))
        
        center_x, center_y = self.width() / 2, self.height() / 2
        
        for glow in self.active_glows:
            glow.draw(painter)
        
        painter.translate(center_x, center_y)
        
        scale_factor = self.beat_scale_factor
        inner_radius = min(self.width(), self.height()) * 0.14 * scale_factor
        
        pen = QPen(Qt.white, 3, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        
        for i in range(self.NUM_BARS):
            angle_rad = math.radians((i / self.NUM_BARS) * 360.0 - 90)
            bar_height = np.clip(self.smoothed_magnitudes[i], 0, 10)**0.65 * self.height() * 0.07 * scale_factor
            
            start_x = inner_radius * math.cos(angle_rad)
            start_y = inner_radius * math.sin(angle_rad)
            end_x = (inner_radius + bar_height) * math.cos(angle_rad)
            end_y = (inner_radius + bar_height) * math.sin(angle_rad)
            
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
        
        painter.setPen(QPen(QColor(200, 220, 255, 200), 4))
        painter.setBrush(QColor(10, 10, 25, 220))
        painter.drawEllipse(QPointF(0, 0), inner_radius, inner_radius)
        
        font = QFont("Arial", int(inner_radius * 0.7))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(220, 220, 255, 180))
        text_rect = QRectF(-inner_radius, -inner_radius, inner_radius*2, inner_radius*2)
        painter.drawText(text_rect, Qt.AlignCenter, "♪")
        
        painter.restore()

        if self.flash_alpha > 0:
            painter.fillRect(self.rect(), QColor(255, 255, 255, self.flash_alpha))
        
        painter.end()


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Starfield Audio Visualizer")
        self.setGeometry(100, 100, 900, 900)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        layout = QVBoxLayout(self.central_widget)
        self.visualizer = CircularVisualizerWidget(self)
        layout.addWidget(self.visualizer)
        
        self.audio_thread = None
        self._create_menu_bar()
    
    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        
        open_audio_action = QAction('&Open Audio File...', self)
        open_audio_action.triggered.connect(self.open_audio_file_dialog)
        file_menu.addAction(open_audio_action)
    
    def open_audio_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", 
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*)")
        if file_name: 
            self.start_visualization(file_name)
    
    def start_visualization(self, file_path):
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
            self.audio_thread.wait()
        
        self.audio_thread = AudioThread(file_path)
        self.audio_thread.data_ready.connect(self.visualizer.update_visuals)
        self.audio_thread.error_occurred.connect(
            lambda msg: self.statusBar().showMessage(msg, 5000))
        self.audio_thread.start()
    
    def closeEvent(self, event):
        """Ensure the audio thread is stopped when the window closes."""
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
