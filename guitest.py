import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

app = customtkinter.CTk()
app.geometry("480x320")
app.title("Coconutz")
app.grid_columnconfigure(2, weight=1)

def button_test():
    print ("Test Successful")

button = customtkinter.CTkButton(master=app, text="Test Button", command=button_test)
button.grid(row=0, column=1, padx=20, pady=20)

app.mainloop()




