import pyaudio
import numpy as np
import tkinter as tk
import sys
import time

# --- AUDIO CONFIGURATION (UPDATE THESE VALUES!) ---
# CRITICAL: Use the index found via 'arecord -l'. Since your device is 'card 0', the index is 0.
INMP441_DEVICE_INDEX = 0        
CHUNK = 1024
FORMAT = pyaudio.paInt32        # INMP441 typically uses 32-bit audio
CHANNELS = 1                    # Start with 1 (Mono). If this fails, try 2.
RATE = 44100                    # A common, reliable sample rate.

# --- PyAudio Setup ---
p = pyaudio.PyAudio()

try:
    # Attempt to open the audio stream with the specified parameters
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
    tk.Label(root, text=f"ERROR: Could not open audio stream.\n\nDetails: {e}\n\nCheck Index, Channels, and Wiring.", 
             fg="red", font=("Helvetica", 12)).pack(padx=20, pady=20)
    root.update()
    root.after(5000, root.destroy) # Close after 5 seconds to show the error
    root.mainloop()
    # Ensure all PyAudio resources are released on error
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
    """Reads audio data, calculates RMS volume, and updates the GUI."""
    try:
        # 1. Read a chunk of data
        # exception_on_overflow=False is important on the RPi
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # 2. Convert raw data (bytes) to NumPy array (32-bit integers)
        audio_data = np.frombuffer(data, dtype=np.int32)
        
        # 3. Calculate Root Mean Square (RMS) volume level
        rms = np.sqrt(np.mean(audio_data**2))
        
        # 4. Scale the RMS value for a readable display (adjust the divisor as needed)
        # Using 10000 provides a reasonable range for typical 32-bit audio input
        scaled_volume = int(rms / 10000)
        
        # 5. Update the UI label and color
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
            print("PyAudio buffer overflow (dropped frames).")
        else:
            print(f"IOError: {e}")

    except Exception as e:
        print(f"An unexpected error occurred during reading: {e}")
        # Stop further updates if a critical error occurs
        return

    # Schedule the function to run again after 50 milliseconds
    root.after(50, update_volume)

# Start the periodic update process
update_volume()

# Start the Tkinter main loop (this keeps the GUI running)
root.mainloop()

# --- Cleanup (Runs after the GUI is closed) ---
print("Closing stream and PyAudio...")
if 'stream' in locals() and stream.is_active():
    stream.stop_stream()
    stream.close()
p.terminate()