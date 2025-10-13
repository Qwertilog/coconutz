import os
import threading
import queue
import numpy as np
import customtkinter as ctk
import pyaudio
import wave
import librosa
import scipy.fft as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, filtfilt

# -------------------------
# Audio parameters
# -------------------------
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2
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
# Find available input device (for Raspberry Pi)
# -------------------------
def get_valid_input_device(p):
    print("\n=== Audio Devices ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"{i}: {info['name']} (inputs: {info['maxInputChannels']}, rate: {info['defaultSampleRate']})")
        if info["maxInputChannels"] > 0:
            return i, int(info["defaultSampleRate"])
    raise RuntimeError("No valid audio input device found!")

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
        ctk.CTkLabel(self, text="2 second audio clip", font=("Arial", 12)).pack(pady=2)
        self.record_button = ctk.CTkButton(self, text="Record", command=self.start_recording)
        self.record_button.pack(pady=20)

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.record_button.configure(fg_color="red", hover_color="darkred")
            self.status_label.configure(text="Status: Recording...")
            threading.Thread(target=self.record_audio_worker, daemon=True).start()

    def record_audio_worker(self):
        audio = pyaudio.PyAudio()
        try:
            device_index, valid_rate = get_valid_input_device(audio)
            print(f"\nUsing input device {device_index} at {valid_rate} Hz\n")

            stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=valid_rate,
                                input=True,
                                frames_per_buffer=CHUNK,
                                input_device_index=device_index)

            frames = [stream.read(CHUNK) for _ in range(int(valid_rate / CHUNK * RECORD_SECONDS))]
            stream.stop_stream()
            stream.close()

            i = 1
            while os.path.isfile(os.path.join(FOLDER, f"recording{i}.wav")):
                i += 1
            filepath = os.path.join(FOLDER, f"recording{i}.wav")

            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(valid_rate)
                wf.writeframes(b''.join(frames))

            gui_queue.put({"type": "record_done", "filepath": filepath})

        except Exception as e:
            print("Recording error:", e)
            self.status_label.configure(text=f"Error: {e}")

        finally:
            audio.terminate()

    def update_status(self, filepath):
        self.recording = False
        self.record_button.configure(fg_color="#1F6AA5", hover_color="#144870")
        self.status_label.configure(text=f"Saved: {filepath}")

# -------------------------
# SpectrumPage
# -------------------------
class SpectrumPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.filepath = None

        ctk.CTkLabel(self, text="Spectrum Analysis", font=("Arial", 16)).pack(pady=5)
        self.info_label = ctk.CTkLabel(self, text="", font=("Arial", 12))
        self.info_label.pack(pady=2)

        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.plot_frame.bind("<Configure>", self.on_resize)

    def update_spectrum(self, filepath):
        self.filepath = filepath
        self.info_label.configure(text=f"Audio Path: {filepath}")
        self.plot_spectrum()

    def plot_spectrum(self):
        if not self.filepath:
            return

        signal, sr = librosa.load(self.filepath, sr=None)
        signal = signal - np.mean(signal)
        signal = low_pass_filter(signal, sr, cutoff=1000)

        ft = sp.fft(signal)
        magnitude = np.abs(ft)
        frequency = np.linspace(0, sr, len(magnitude))
        half = len(frequency) // 2

        self.ax.clear()
        self.ax.plot(frequency[1:half], magnitude[1:half])
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude")
        self.ax.set_title("FFT Spectrum (Low-Pass 1kHz)")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def on_resize(self, event):
        width, height = event.width, event.height
        dpi = self.fig.get_dpi()
        self.fig.set_size_inches(width / dpi, height / dpi)
        self.canvas.draw_idle()

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app = SpectrumAnalyzerApp()
    app.mainloop()
