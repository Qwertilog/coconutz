import pyaudio
import numpy as np
import tkinter as tk
import sys
import time

# --- AUDIO CONFIGURATION (Most Robust Settings for INMP441/PyAudio) ---
# We are switching to 16-bit Mono, which is less likely to suffer from 32-bit data corruption.
INMP441_DEVICE_INDEX = 0        # Confirmed index (Card 0)
CHUNK = 1024
FORMAT = pyaudio.paInt16        # CHANGED to 16-bit format (paInt16) for stability
CHANNELS = 1                    # CHANGED to 1 (Mono)
RATE = 44100                    # Set to a common, standard sample rate

# --- PyAudio Setup ---
p = pyaudio.PyAudio()

try:
    # Attempt to open the audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=INMP441_DEVICE_INDEX)
    print(f"✅ Opened audio stream on device index {INMP441_DEVICE_INDEX} at {RATE} Hz.")

except Exception as e:
    # Display error in a simple UI window and exit gracefully
    print(f"❌ Error opening audio stream: {e}")
    root = tk.Tk()
    root.title("Mic Check Error")
    tk.Label(root, text=f"ERROR: Could not open audio stream.\n\nDetails: {e}", 
             fg="red", font=("Helvetica", 12)).pack(padx=20, pady=20)
    root.update()
    root.after(5000, root.destroy)
    root.mainloop()
    p.terminate()
    sys.exit()


# --- Tkinter UI Setup ---
root = tk.Tk()
root.title("INMP441 Real-Time Volume Check")
root.geometry("400x200")

# Label for the volume indicator
volume_label = tk.Label(root, text="Volume: ---", font=("Helvetica", 36, "bold"), pady=10)
volume_label.pack(expand=True)

# Status line for configuration details
status_label = tk.Label(root, text=f"Index: {INMP441_DEVICE_INDEX} | Channels: {CHANNELS} | Rate: {RATE} Hz", font=("Helvetica", 10), fg="gray")
status_label.pack(pady=(0, 10))

# Function to calculate and update volume
def update_volume():
    """Reads audio data, calculates RMS volume on the first channel, and updates the GUI."""
    try:
        # 1. Read a chunk of data
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # 2. Convert raw data (bytes) to NumPy array
        # CRITICAL CHANGE: Use '<i2' for 16-bit little-endian integer to match paInt16
        audio_data = np.frombuffer(data, dtype='<i2') 
        
        # 3. No slicing needed since CHANNELS=1
        left_channel_data = audio_data
        
        # 4. Calculate Root Mean Square (RMS) volume level on the valid channel
        if left_channel_data.size == 0:
            rms = 0
        else:
            rms = np.sqrt(np.mean(left_channel_data**2))
        
        # 5. Scale the RMS value for a readable display
        # The divisor is significantly smaller for 16-bit audio (max value is 32767).
        scaled_volume = int(rms / 300)
        
        # 6. Update the UI label and color