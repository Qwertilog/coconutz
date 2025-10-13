import customtkinter as ctk

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Page App")
        self.geometry("800x600")

        # Main content frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Navigation buttons (example)
        nav_frame = ctk.CTkFrame(self)
        nav_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        login_button = ctk.CTkButton(nav_frame, text="Login", command=self.show_login_page)
        login_button.pack(pady=5)
        
        signup_button = ctk.CTkButton(nav_frame, text="Signup", command=self.show_signup_page)
        signup_button.pack(pady=5)

        # Initially show the login page
        self.show_login_page()

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_login_page(self):
        self.clear_main_frame()
        # Create and pack/place login page widgets into self.main_frame
        login_label = ctk.CTkLabel(self.main_frame, text="Login Page Content")
        login_label.pack(pady=20)
        # ... add other login widgets ...

    def show_signup_page(self):
        self.clear_main_frame()
        # Create and pack/place signup page widgets into self.main_frame
        signup_label = ctk.CTkLabel(self.main_frame, text="Signup Page Content")
        signup_label.pack(pady=20)
        # ... add other signup widgets ...

if __name__ == "__main__":
    app = App()
    app.mainloop()