import pyaudio
import numpy as np
import tkinter as tk
import time

# --- Audio Configuration (MATCH THIS TO YOUR INMP441 SETUP) ---
# IMPORTANT: Replace 1 with the actual PyAudio input device index for your INMP441
INMP441_DEVICE_INDEX = 1 
CHUNK = 1024
FORMAT = pyaudio.paInt32        # INMP441 typically uses 32-bit audio
CHANNELS = 2                    # 1 for mono, 2 for stereo (based on your wiring)
RATE = 48000                    # Common sample rate for INMP441

# --- PyAudio Setup ---
p = pyaudio.PyAudio()

try:
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=INMP441_DEVICE_INDEX)
    print(f"Opened audio stream on device index {INMP441_DEVICE_INDEX}")

except Exception as e:
    print(f"Error opening audio stream: {e}")
    # Display error and exit if audio stream fails
    root = tk.Tk()
    root.title("Mic Check Error")
    tk.Label(root, text=f"ERROR: Could not open audio stream.\nCheck device index and setup.\n{e}", fg="red", font=("Helvetica", 12)).pack(padx=20, pady=20)
    root.mainloop()
    exit()

# --- Tkinter UI Setup ---
root = tk.Tk()
root.title("INMP441 Mic Checker")
root.geometry("400x200")

# Styling for the volume indicator
volume_label = tk.Label(root, text="Volume: ---", font=("Helvetica", 24, "bold"), pady=10)
volume_label.pack(expand=True)

status_label = tk.Label(root, text=f"Listening on Channel(s): {CHANNELS}, Rate: {RATE} Hz", font=("Helvetica", 10), fg="gray")
status_label.pack(pady=(0, 10))

# Function to calculate and update volume
def update_volume():
    try:
        # Read a chunk of data from the stream
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Convert raw audio data (bytes) to NumPy array for calculation
        # np.frombuffer interprets the raw data as 32-bit integers
        audio_data = np.frombuffer(data, dtype=np.int32)
        
        # Calculate the Root Mean Square (RMS) volume level
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Scale the RMS value (a raw digital value) to a more readable number (e.g., 0 to 10000)
        # Max theoretical 32-bit value is 2,147,483,647. We'll use a smaller scale for display.
        scaled_volume = int(rms / 10000)
        
        # Update the UI label
        volume_label.config(text=f"Volume: {scaled_volume}")
        
        # Change color based on volume (simple visual feedback)
        if scaled_volume > 150:
            volume_label.config(fg="red")
        elif scaled_volume > 50:
            volume_label.config(fg="orange")
        else:
            volume_label.config(fg="green")

    except IOError as e:
        # Handle the common PyAudio error (e.g., buffer overflow)
        if e.errno == -9988:
            print("PyAudio buffer overflow (dropped frames). Continuing...")
        else:
            print(f"IOError: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Schedule the function to run again after a small delay (e.g., 50ms)
    root.after(50, update_volume)

# Start the periodic update process
update_volume()

# Start the Tkinter main loop
root.mainloop()

# --- Cleanup ---
print("Closing stream and PyAudio...")
if 'stream' in locals() and stream.is_active():
    stream.stop_stream()
    stream.close()
p.terminate()