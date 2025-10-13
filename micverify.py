import pyaudio

p = pyaudio.PyAudio()
print("--- PyAudio Device List ---")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info.get('maxInputChannels') > 0:
        print(f"Index {i}: {device_info.get('name')}")
        print(f"  Max Input Channels: {device_info.get('maxInputChannels')}")
        print(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
print("----------------------------")
p.terminate()