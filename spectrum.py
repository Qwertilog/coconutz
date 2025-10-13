import os
import numpy as np
import customtkinter
import pyaudio
import wave
import threading
import scipy.fft as sp
import librosa
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 1

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

class SpectrumAnalyzer(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("480x320")
        self.resizable(False, False)
        self.title("Spectrum Analyzer")

        self.mainframe = customtkinter.CTkFrame(self)
        self.mainframe.pack(fill="both", expand=True)
        self.mainframe.grid_rowconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(0, weight=1)

        self.pages = {}
        for Page in (MainPage, SpectrumPage):
            page = Page(self.mainframe, self)
            self.pages[Page.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_mainpage()

    def show_mainpage(self):
        self.pages["MainPage"].tkraise()

    def show_spectrumpage(self):
        mainpage = self.pages["MainPage"]
        spectrum_page = self.pages["SpectrumPage"]
        if hasattr(mainpage, "filepath"):
            spectrum_page.update_filepath(mainpage.filepath)
        spectrum_page.tkraise()


class MainPage(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.recording = False

        mainlabel = customtkinter.CTkLabel(self, text="Click the record button to start", font=("Arial", 24))
        mainlabel.pack(pady=20)
        self.status_label = customtkinter.CTkLabel(self, text="Status: Not Recording", font=("Arial", 16))
        self.status_label.pack(pady=10)
        info_label = customtkinter.CTkLabel(self, text="1 second audio clip", font=("Arial", 12))
        info_label.pack(pady=1)
        self.record_button = customtkinter.CTkButton(self, text="Record", command=self.record_audio)
        self.record_button.pack(pady=20)

    def record_audio(self):
        if not self.recording:
            self.recording = True
            self.record_button.configure(fg_color="red", hover_color="darkred")
            self.status_label.configure(text="Status: Recording...")
            threading.Thread(target=self._record, daemon=True).start()

    def _record(self):
        folder = "recordings"
        os.makedirs(folder, exist_ok=True)

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
        stream.stop_stream()
        stream.close()
        audio.terminate()

        i = 1
        while os.path.isfile(os.path.join(folder, f"recording{i}.wav")):
            i += 1
        self.filepath = os.path.join(folder, f"recording{i}.wav")

        with wave.open(self.filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        self.recording = False
        if self.winfo_exists():
            self.after(0, lambda: [
            self.record_button.configure(fg_color="#1F6AA5", hover_color="#144870"),
            self.status_label.configure(text=f"Saved: {self.filepath}"),
            self.controller.show_spectrumpage()
        ])


class SpectrumPage(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.filepath = None

        self.status_label = customtkinter.CTkLabel(self, text="Spectrum Analysis", font=("Arial", 16))
        self.status_label.pack(pady=1)
        self.info_label = customtkinter.CTkLabel(self, text=f"Audio Path: {self.filepath}", font=("Arial", 12))
        self.info_label.pack(pady=1)

        self.plot_frame = customtkinter.CTkFrame(self)
        self.plot_frame.pack(fill="both", expand=True, pady=10, padx=10)

    def update_filepath(self, filepath):
        self.filepath = filepath
        if not self.winfo_exists(): 
            return
        self.info_label.configure(text=f"Audio Path: {self.filepath}")
        self.after(0, self.analyze_audio)

    def analyze_audio(self):
        if not self.filepath or not self.winfo_exists():
            return

        signal, sr = librosa.load(self.filepath, sr=None)
        ft = sp.fft(signal)
        magnitude = np.abs(ft)
        frequency = np.linspace(0, sr, len(magnitude))
        half = len(frequency)//2

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(frequency[:half], magnitude[:half])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title('FFT Spectrum')

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = SpectrumAnalyzer()
    app.mainloop()
