import os
import threading
import queue
import numpy as np
import customtkinter as ctk
# import pyaudio # REMOVED: Replaced by pvrecorder
import pvrecorder # ADDED: For I2S microphone access
import wave
import librosa
import scipy.fft as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, filtfilt

# -------------------------
# Audio parameters (CRITICAL: Adjusted for INMP441 I2S Mic)
# -------------------------
# Use a simple flag for format, as pvrecorder handles the details.
# The INMP441 driver often performs best at 48000 Hz with 32-bit internal data.
CHANNELS = 1
RATE = 48000
CHUNK = 1024 # Note: pvrecorder uses frame_length, set to 1024 frames/buffer
RECORD_SECONDS = 2
DEVICE_INDEX = 2 # CRITICAL: Set this to the working index (1 or 2, based on troubleshooting)

FOLDER = "recordings"
os.makedirs(FOLDER, exist_ok=True)

# -------------------------
# Tkinter setup
# -------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# -------------------------
# Thread-safe queue
# -------------------------
gui_queue = queue.Queue()

# -------------------------
# Low-pass filter
# -------------------------
def low_pass_filter(signal, sr, cutoff=1000):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# -------------------------
# Main App
# -------------------------
class SpectrumAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Spectrum Analyzer")
        self.geometry("480x320")
        self.resizable(False, False)

        # Container frame
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Pages
        self.pages = {}
        self.pages["MainPage"] = MainPage(self.container, self)
        self.pages["SpectrumPage"] = SpectrumPage(self.container, self)
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

        self.show_page("MainPage")

        # Start main queue loop
        self.after(100, self.process_queue)

    def show_page(self, page_name):
        self.pages[page_name].tkraise()

    def process_queue(self):
        while not gui_queue.empty():
            try:
                msg = gui_queue.get_nowait()
            except queue.Empty:
                break
            if msg["type"] == "record_done":
                filepath = msg["filepath"]
                self.pages["MainPage"].update_status(filepath)
                self.pages["SpectrumPage"].update_spectrum(filepath)
                self.show_page("SpectrumPage")
        self.after(100, self.process_queue)

# -------------------------
# MainPage
# -------------------------
class MainPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.recording = False
        self.filepath = None

        ctk.CTkLabel(self, text="Click Record to start", font=("Arial", 20)).pack(pady=15)
        self.status_label = ctk.CTkLabel(self, text="Status: Not Recording", font=("Arial", 14))
        self.status_label.pack(pady=10)
        ctk.CTkLabel(self, text=f"{RECORD_SECONDS} second audio clip at {RATE} Hz", font=("Arial", 12)).pack(pady=2)
        self.record_button = ctk.CTkButton(self, text="Record", command=self.start_recording)
        self.record_button.pack(pady=20)

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.record_button.configure(fg_color="red", hover_color="darkred")
            self.status_label.configure(text="Status: Recording...")
            threading.Thread(target=self.record_audio_worker, daemon=True).start()

    def record_audio_worker(self):
        # *** PvRecorder Implementation ***
        frames = []
        try:
            # Initialize pvrecorder
            recorder = pvrecorder.PvRecorder(
                frame_length=CHUNK,
                device_index=DEVICE_INDEX)
            
            # pvrecorder uses an internal format that is determined by the driver,
            # which for INMP441 is 32-bit. We use a total number of frames.
            total_frames = int(RATE / CHUNK * RECORD_SECONDS)
            
            recorder.start()
            
            for _ in range(total_frames):
                # pvrecorder returns a list of signed 32-bit integers
                frame = recorder.read()
                # Convert the list of integers to a 32-bit bytes object
                # and append to frames list for saving.
                frames.append(np.array(frame, dtype=np.int32).tobytes())
            
            recorder.stop()
            recorder.delete()
            
        except Exception as e:
            print(f"PvRecorder Error: {e}")
            gui_queue.put({"type": "record_done", "filepath": "ERROR"})
            return

        # *** Save WAV File ***
        i = 1
        while os.path.isfile(os.path.join(FOLDER, f"recording{i}.wav")):
            i += 1
        filepath = os.path.join(FOLDER, f"recording{i}.wav")

        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            # CRITICAL: We save the file as 32-bit (4 bytes) to match the INMP441 driver
            wf.setsampwidth(4) # 4 bytes = 32-bit
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        gui_queue.put({"type": "record_done", "filepath": filepath})

    def update_status(self, filepath):
        self.recording = False
        self.record_button.configure(fg_color="#1F6AA5", hover_color="#144870")
        if filepath == "ERROR":
            self.status_label.configure(text="Status: Recording Failed!", fg_color="red")
        else:
            self.status_label.configure(text=f"Saved: {filepath}", fg_color="transparent")

# -------------------------
# SpectrumPage (Responsive)
# -------------------------
class SpectrumPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.filepath = None

        # Labels
        ctk.CTkLabel(self, text="Spectrum Analysis", font=("Arial", 16)).pack(pady=5)
        self.info_label = ctk.CTkLabel(self, text="", font=("Arial", 12))
        self.info_label.pack(pady=2)

        # Plot frame
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create initial empty figure and canvas
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Bind resize event
        self.plot_frame.bind("<Configure>", self.on_resize)
        
        # Back button
        ctk.CTkButton(self, text="New Recording", command=lambda: controller.show_page("MainPage")).pack(pady=5)


    def update_spectrum(self, filepath):
        self.filepath = filepath
        self.info_label.configure(text=f"Audio Path: {filepath}")
        self.plot_spectrum()

    def plot_spectrum(self):
        if not self.filepath or self.filepath == "ERROR":
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Error: No Audio Data", ha='center', va='center', fontsize=14)
            self.canvas.draw_idle()
            return

        # Load audio (Librosa handles the 32-bit WAV automatically)
        signal, sr = librosa.load(self.filepath, sr=None)

        # Remove DC
        signal = signal - np.mean(signal)

        # Apply low-pass filter (1 kHz to include musical notes)
        signal = low_pass_filter(signal, sr, cutoff=1000)

        # FFT
        ft = sp.fft(signal)
        magnitude = np.abs(ft)
        frequency = np.linspace(0, sr, len(magnitude))
        half = len(frequency) // 2

        # Clear and plot
        self.ax.clear()
        self.ax.plot(frequency[1:half], magnitude[1:half])
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude")
        self.ax.set_title("FFT Spectrum (Low-Pass 1kHz)")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def on_resize(self, event):
        # This function ensures the plot resizes correctly within the frame.
        width, height = event.width, event.height
        dpi = self.fig.get_dpi()
        # Subtract some padding to prevent a tight fit causing scrollbars
        self.fig.set_size_inches((width-10)/dpi, (height-10)/dpi) 
        self.fig.tight_layout()
        self.canvas.draw_idle()

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app = SpectrumAnalyzerApp()
    app.mainloop()