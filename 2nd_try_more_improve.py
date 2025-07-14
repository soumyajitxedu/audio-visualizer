import sys
import numpy as np
import sounddevice as sd
import wave
import time
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QAction, QFileDialog, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QImage, QFont

# --- Audio Processing Thread ---
# This thread handles loading and processing the audio file to not freeze the GUI.
class AudioThread(QThread):
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_path, chunk_size=2048):
        super().__init__()
        self.chunk_size = chunk_size
        self.file_path = file_path
        self.running = True

    def run(self):
        try:
            with wave.open(self.file_path, 'rb') as wf:
                self.sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)

            if n_channels > 1:
                audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)

            audio_data_float = audio_data.astype(np.float32) / (2**15)
            
            sd.play(audio_data_float, self.sample_rate)

            num_chunks = len(audio_data_float) // self.chunk_size
            start_time = time.time()

            for i in range(num_chunks):
                if not self.running: break

                expected_time = start_time + (i * self.chunk_size / self.sample_rate)
                while time.time() < expected_time:
                    time.sleep(0.001)

                chunk = audio_data_float[i * self.chunk_size:(i + 1) * self.chunk_size]
                if chunk.size == 0: continue
                
                # --- FFT Analysis ---
                fft_data = np.fft.rfft(chunk * np.hanning(len(chunk))) # Use a window function
                fft_freq = np.fft.rfftfreq(len(chunk), 1.0 / self.sample_rate)
                magnitudes = np.abs(fft_data)

                # Overall loudness for pulsing effect
                loudness = np.sqrt(np.mean(chunk**2))

                self.data_ready.emit({
                    'magnitudes': magnitudes,
                    'fft_freq': fft_freq,
                    'loudness': loudness,
                })

        except Exception as e:
            self.error_occurred.emit(f"Audio thread error: {e}")
        finally:
            sd.stop()

    def stop(self):
        self.running = False
        sd.stop()


# --- The Visualizer Widget ---
# This widget uses QPainter to draw the visualizer.
class CircularVisualizerWidget(QWidget):
    # --- Configurable Parameters ---
    NUM_BARS = 120 # Number of bars in the circle
    MIN_FREQ = 20 # Hz
    MAX_FREQ = 22000 # Hz
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.background_image = None
        
        # Audio data state
        self.magnitudes = np.zeros(self.NUM_BARS)
        self.loudness = 0.0
        
        # Smoothed values for fluid animation
        self.smoothed_magnitudes = np.zeros(self.NUM_BARS)
        self.smoothed_loudness = 0.0
        
        # Define logarithmic frequency bins
        self.log_freq_indices = self.create_log_freq_indices()

        # Timer for smooth animation (60 FPS)
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(1000 // 60)
        self.render_timer.timeout.connect(self._update_frame)
        self.render_timer.start()

    def set_background(self, image_path):
        """Loads and sets the background image."""
        self.background_image = QImage(image_path)
        self.update()

    def create_log_freq_indices(self):
        """Creates logarithmically spaced frequency bins for the bars."""
        freqs = np.logspace(np.log10(self.MIN_FREQ), np.log10(self.MAX_FREQ), self.NUM_BARS + 1)
        return freqs

    def update_visuals(self, audio_data):
        """Slot to receive new audio data from the AudioThread."""
        raw_mags = audio_data['magnitudes']
        raw_freqs = audio_data['fft_freq']
        self.loudness = audio_data['loudness']
        
        # Process FFT data into logarithmic bins
        processed_mags = np.zeros(self.NUM_BARS)
        for i in range(self.NUM_BARS):
            min_f = self.log_freq_indices[i]
            max_f = self.log_freq_indices[i+1]
            
            # Find indices of FFT frequencies that fall into the current bin
            indices = np.where((raw_freqs >= min_f) & (raw_freqs < max_f))
            if indices[0].size > 0:
                # Average the magnitudes in the bin
                processed_mags[i] = np.mean(raw_mags[indices])
            else:
                processed_mags[i] = 0
                
        self.magnitudes = processed_mags

    def _update_frame(self):
        """Called by the timer to update smoothed values and schedule a repaint."""
        # Smoothing for fluid animations
        self.smoothed_magnitudes = self.smoothed_magnitudes * 0.85 + self.magnitudes * 0.15
        self.smoothed_loudness = self.smoothed_loudness * 0.9 + self.loudness * 0.1
        self.update() # Triggers the paintEvent

    def paintEvent(self, event):
        """The main drawing function, called every time the widget updates."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background with black
        painter.fillRect(self.rect(), Qt.black)
        
        # Draw background image if it exists
        if self.background_image:
            painter.drawImage(self.rect(), self.background_image)

        # Move origin to the center of the widget
        center_x, center_y = self.width() / 2, self.height() / 2
        painter.translate(center_x, center_y)

        # --- Draw the Visualizer ---
        # Pulse effect for the entire visualizer's scale
        scale_factor = 1.0 + self.smoothed_loudness * 1.5
        
        # Central circle
        inner_radius = min(self.width(), self.height()) * 0.15 * scale_factor
        
        # Draw the bars
        pen = QPen(Qt.white, 3, Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)

        for i in range(self.NUM_BARS):
            angle_deg = (i / self.NUM_BARS) * 360.0
            angle_rad = math.radians(angle_deg - 90) # Subtract 90 to start at the top
            
            magnitude = self.smoothed_magnitudes[i]
            
            # Scale magnitude and apply a power curve to enhance visual response
            bar_height = (magnitude**0.8) * self.height() * 0.05 * scale_factor
            
            # Define start and end points of the bar
            start_x = inner_radius * math.cos(angle_rad)
            start_y = inner_radius * math.sin(angle_rad)
            end_x = (inner_radius + bar_height) * math.cos(angle_rad)
            end_y = (inner_radius + bar_height) * math.sin(angle_rad)
            
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
            
        # Draw the central circle on top
        painter.setPen(QPen(QColor(200, 200, 255), 4))
        painter.setBrush(QColor(10, 10, 25, 200))
        painter.drawEllipse(int(-inner_radius), int(-inner_radius), int(inner_radius * 2), int(inner_radius * 2))

        # Optional: Draw a letter in the middle like the 'S' logo
        font = QFont("Arial", int(inner_radius * 0.8))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(220, 220, 255, 150))
        
        # =========================================================
        # THE FIX IS HERE: Casting the float values to integers.
        # =========================================================
        painter.drawText(self.rect().translated(int(-center_x), int(-center_y)), Qt.AlignCenter, "S")


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Circular Audio Visualizer")
        self.setGeometry(100, 100, 800, 800)
        
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
        
        open_audio_action = QAction('&Open WAV File...', self)
        open_audio_action.triggered.connect(self.open_wav_file_dialog)
        file_menu.addAction(open_audio_action)
        
        open_image_action = QAction('Set &Background Image...', self)
        open_image_action.triggered.connect(self.open_image_dialog)
        file_menu.addAction(open_image_action)

    def open_wav_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV Files (*.wav)")
        if file_name:
            self.start_visualization(file_name)

    def open_image_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.visualizer.set_background(file_name)

    def start_visualization(self, file_path):
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop(); self.audio_thread.wait()
        
        self.audio_thread = AudioThread(file_path)
        self.audio_thread.data_ready.connect(self.visualizer.update_visuals)
        self.audio_thread.error_occurred.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
        self.audio_thread.start()

    def closeEvent(self, event):
        if self.audio_thread:
            self.audio_thread.stop(); self.audio_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
