import pvrecorder
import numpy as np
import tkinter as tk
import sys
import time

# --- AUDIO CONFIGURATION (MATCHING SUCCESSFUL arecord COMMAND) ---
# CRITICAL: Device Index changed to 2 as requested.
DEVICE_INDEX = 2 

# Match the settings from the successful arecord command: 48000 Hz, 32-bit data
FRAME_LENGTH = 1024  # Increased frame length for better 48kHz reading
RATE = 48000
FORMAT_BITS = 32

# --- PvRecorder Setup ---
recorder = None
try:
    # Initialize the recorder
    recorder = pvrecorder.PvRecorder(
        frame_length=FRAME_LENGTH,
        device_index=DEVICE_INDEX)
    
    recorder.start()
    print(f"✅ Opened audio stream on device index {DEVICE_INDEX} at {RATE} Hz.")

except Exception as e:
    # Display error in a simple UI window and exit gracefully
    print(f"❌ Error initializing PvRecorder: {e}")
    root = tk.Tk()
    root.title("Mic Check Error")
    tk.Label(root, text=f"ERROR: Could not initialize PvRecorder.\n\nDetails: {e}", 
             fg="red", font=("Helvetica", 12)).pack(padx=20, pady=20)
    root.update()
    root.after(5000, root.destroy)
    root.mainloop()
    sys.exit()

# --- Tkinter UI Setup ---
root = tk.Tk()
root.title("INMP441 Real-Time Volume Check (pvrecorder)")
root.geometry("400x200")

# Label for the volume indicator
volume_label = tk.Label(root, text="Volume: ---", font=("Helvetica", 36, "bold"), pady=10)
volume_label.pack(expand=True)

# Status line for configuration details
status_label = tk.Label(root, text=f"Index: {DEVICE_INDEX} | Rate: {RATE} Hz | Format: {FORMAT_BITS}-bit", font=("Helvetica", 10), fg="gray")
status_label.pack(pady=(0, 10))

# Function to calculate and update volume
def update_volume():
    """Reads audio data, calculates RMS volume, and updates the GUI."""
    try:
        # 1. Read a frame of data
        frame = recorder.read()
        
        # 2. CRITICAL CHANGE: Convert to NumPy array using 32-bit integers
        # This correctly interprets the S32_LE data from the I2S driver.
        audio_data = np.array(frame, dtype=np.int32) 
        
        # 3. Calculate Root Mean Square (RMS) volume level
        if audio_data.size == 0:
            rms = 0
        else:
            rms = np.sqrt(np.mean(audio_data**2))
        
        # 4. Scale the RMS value for a readable display
        # Use a divisor appropriate for 32-bit audio (Max RMS ~2.1 billion).
        scaled_volume = int(rms / 1000000) 
        
        # 5. Update the UI label and color
        volume_label.config(text=f"Volume: {scaled_volume}")
        
        if scaled_volume > 100:
            volume_label.config(fg="red")     # Loud
        elif scaled_volume > 20:
            volume_label.config(fg="orange")  # Medium
        else:
            volume_label.config(fg="green")   # Quiet/Ambient

    except Exception as e:
        print(f"An unexpected error occurred during reading: {e}")
        # Stop further updates if a critical error occurs
        return

    # Schedule the function to run again after 50 milliseconds
    root.after(50, update_volume)

# Start the periodic update process
update_volume()

# Start the Tkinter main loop
root.mainloop()

# --- Cleanup (Runs after the GUI is closed) ---
print("Closing recorder...")
if recorder is not None:
    recorder.stop()
    recorder.delete()