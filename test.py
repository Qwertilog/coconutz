# --- Record Function ---
    def record_audio(self):
        if not self.recording:
            self.recording = True
            self.record_button.configure(fg_color="red", hover_color="darkred")
            self.status_label.configure(text="Status: Recording...")
            threading.Thread(target=self._record).start()

    def _record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, 
                            channels=CHANNELS,
                            rate=RATE,
                            input=True, 
                            frames_per_buffer=CHUNK)
        frames = []
        num_chunks = int(RATE / CHUNK * RECORD_SECONDS)

        for _ in range(num_chunks):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        exists  = True
        i = 1
        while exists:
            if os.path.isfile(f"recording{i}.wav"):
                i += 1
            else:
                exists = False

        wf = wave.open(f"recording{i}.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.recording = False
        self.record_button.configure(fg_color="#1F6AA5", hover_color="#144870")
        self.status_label.configure(text="Saved: {filename}")

        self.analyze_audio(f"recording{i}.wav")

# --- Spectrum Analysis Page ---
    def create_spectrum_page(self):
        self.spectrumlabel = customtkinter.CTkLabel(master=self.spectrum_frame, text="Spectrum Analysis", font=("Arial", 24))
        self.spectrumlabel.pack(pady=20)
        self.back_button = customtkinter.CTkButton(master=self.spectrum_frame, text="Back", command=lambda: self.show_frame(self.mainframe))
        self.back_button.pack(pady=10)

    def analyze_audio(self, filename):
        y, sr = librosa.load(filename, sr=None)
        spectrum = np.abs(librosa.stft(y))
        spectrum_db = librosa.amplitude_to_db(spectrum, ref=np.max)

    


        print(f"\nAnalyzed: {filename}")
        print(f"Sample rate: {sr} Hz")
        print(f"Spectrogram shape: {spectrum_db.shape}")