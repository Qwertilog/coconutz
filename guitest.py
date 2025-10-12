import customtkinter
import RPi.GPIO as gpio
from time import sleep

gpio.setwarnings(False)
gpio.setmode(gpio.BCM)
gpio.setup(36, gpio.OUT)

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

app = customtkinter.CTk()
app.geometry("480x320")
app.title("Coconutz")
app.grid_columnconfigure(2, weight=1)

status_label = customtkinter.CTkLabel(master=app, text="Relay off", font=("Arial", 16))
status_label.grid(row=0, column=2, padx=20, pady=20)

def turn_on():
    gpio.output(36, gpio.HIGH)
    status_label.configure(text="Relay on")

def turn_off():
    gpio.output(36, gpio.LOW)
    status_label.configure(text="Relay off")

button = customtkinter.CTkButton(master=app, text="Turn on", command=turn_on)
button.grid(row=0, column=1, padx=20, pady=20)
button = customtkinter.CTkButton(master=app, text="Turn off", command=turn_off)
button.grid(row=0, column=3, padx=20, pady=20)

def on_close():
    gpio.cleanup()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)
app.mainloop()




