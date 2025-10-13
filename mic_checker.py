import pyaudio
import numpy as np
import tkinter as tk
import sys
import time

#new mic checker
# --- AUDIO CONFIGURATION (Based on successful ALSA tests) ---
INMP441_DEVICE_INDEX = 0        # Confirmed index (Card 0)
CHUNK = 1024
FORMAT = pyaudio.paInt32        # 32-bit format (Standard for INMP441 on RPi)
CHANNELS = 2                    # Confirmed working channel count (Stereo)
RATE = 48000                    # Confirmed actual rate used by the driver

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
        # Using '<i4' specifies little-endian 4-byte (32-bit) integers.
        audio_data = np.frombuffer(data, dtype='<i4')
        
        # 3. CRITICAL: Slice the data to read ONLY the first channel.
        # This prevents the NaN crash from corrupted data in the second channel.
        # The '::2' slicing takes every second sample, which is the left/first channel.
        left_channel_data = audio_data[::2]
        
        # 4. Calculate Root Mean Square (RMS) volume level on the valid channel
        if left_channel_data.size == 0:
            rms = 0
        else:
            rms = np.sqrt(np.mean(left_channel_data**2))
        
        # 5. Scale the RMS value for a readable display
        scaled_volume = int(rms / 10000)
        
        # 6. Update the UI label and color
        volume_label.config(text=f"Volume: {scaled_volume}")
        
        if scaled_volume > 200:
            volume_label.config(fg="red")     # Loud
        elif scaled_volume > 50:
            volume_label.config(fg="orange")  # Medium
        else:
            volume_label.config(fg="green")   # Quiet/Ambient

    except IOError as e:
        # Common error when the audio buffer is slightly slow
        if e.errno == -9988:
            print("Py