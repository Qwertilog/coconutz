from pathlib import Path
from tkinter import Tk, Canvas, Text, Button, PhotoImage, filedialog, messagebox, PhotoImage
import cv2
import numpy as np
import pandas as pd
import sys
import os
import time
import platform
import threading
from joblib import load
from PIL import Image, ImageTk

video_capture = None   
picam = None          
camera_after = None

selected_file = None
selected_label = None

processed_file = None
hsv_class = None

USE_PICAMERA2 = False

try:
    model =load("camera_model.pkl")
except Exception as e:
    print(f"[WARN] Model not loaded: {e}")
    model = None

try: 
    from picamera2 import PiCamera
    USE_PICAMERA2 = True
except ImportError:
    USE_PICAMERA2 = False

def stop_camera():
    global video_capture, picam, camera_after
    try:
        if camera_after is not None:
            window.after_cancel(camera_after)
    except Exception:
        pass
    camera_after = None

    try:
        if video_capture is not None:
            if hasattr(video_capture, "isOpened") and video_capture.isOpened():
                video_capture.release()
            video_capture = None
    except Exception:
        video_capture = None

    try:
        if picam is not None:
            try:
                picam.stop()
            except Exception:
                pass
            picam = None
    except Exception:
        picam = None


if getattr(sys, 'frozen', False):  # Running as EXE
    base_dir = Path(sys.executable).parent
else:  # Running as script
    base_dir = Path(__file__).parent

csv_path = base_dir / "coconut_features.csv"

folder_path1 = base_dir / "Data Collection Captures"
folder_path2 = base_dir / "Data Detection Captures"
os.makedirs(folder_path1, exist_ok=True)
os.makedirs(folder_path2, exist_ok=True)

def switch_page(page_name):
    try:
        stop_camera()
    except NameError:
        pass

    for widget in window.winfo_children():
        if widget is not canvas:
            widget.destroy()

    pages = {
        "main": load_main_page,
        "data_collection1": load_data_collection_page_1,
        "data_collection2": load_data_collection_page_2,
        "data_collection3": load_data_collection_page_3,
        "data_collection4": load_data_collection_page_4,
        "data_detection1": load_data_detection_page_1,
        "data_detection2": load_data_detection_page_2,
        "data_detection3": load_data_detection_page_3,
        "data_detection4": load_data_detection_page_4,
        "data_detection5": load_data_detection_page_5
    }

    if page_name in pages:
        pages[page_name]()


# UI HELP #
def btn(text, x, y, w, h, font_size, command):
    btn = Button(
     window,
     text=text,
     font=("InriaSans Regular", font_size),
     fg="white",
     bg="#C82333",
     activebackground="#A71D2A",
     activeforeground="white",
     borderwidth=0,
    highlightthickness=2,
    highlightbackground="white",
    highlightcolor="white",  
     command=command, 
     relief="flat"
     )
    btn.place(x=x, y=y, width=w, height=h)
    return btn

def btn_hover(button, normal="#C82333", hover="#E74C3C"):
    def on_enter(e): button.config(bg=hover)
    def on_leave(e): button.config(bg=normal)
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

def create_text(content, x, y, w, h, font_size=24):
    txt = Text(
        window,
        bd=0,
        bg="#7571E6",
        fg="#FFFFFF",
        highlightthickness=0,
        font=("InriaSans Regular", font_size * -1),
    )
    txt.insert("1.0", content)
    txt.tag_configure("center", justify="center")
    txt.tag_add("center", "1.0", "end")
    txt.place(x=x, y=y, width=w, height=h)
    return txt

def camera_prepro(image_path):
    def task():
        global processed_file, hsv_class

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Unreadable image")
            
            hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower=np.array([5, 30, 30])
            upper=np.array([90, 255, 255])

            mask=cv2.inRange(hsv, lower, upper)
            kernel=np.ones((3, 3), np.uint8)
            mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            masked_img=cv2.bitwise_and(img, img, mask=mask)
            resized_img=cv2.resize(masked_img, (224, 224))

            processed_file=os.path.splitext(image_path)[0] + "_processed.jpg"
            cv2.imwrite(processed_file, resized_img)

            features= camera_features(processed_file)

            if model is not None:
                try:
                    x = np.array([features])
                    probs = model.predict_proba(x)[0]
                    class_labels = model.classes_
                    hsv_class ={class_labels[i]: float(probs[i]) for i in range(len(class_labels))}

                except Exception:
                    hsv_class = None

            else:
                print("[INFO] No model — only preprocessing done.")

        except Exception as e:
            print(f"[ERROR] Preprocessing/classification failed: {e}")

    threading.Thread(target=task).start()

def camera_features(image_path):
    img=cv2.imread(image_path)
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    return [h_mean, s_mean, v_mean]


def save_features_to_csv(features, filepath, label):
    df = pd.DataFrame(
        [[label] + features],
        columns=["Label", "H_mean", "S_mean", "V_mean"]
    )
    df.to_csv(filepath, mode="a", header=not Path(filepath).exists(), index=False)

# MAIN WINDOW #
window = Tk()
window.geometry("387x308")
window.configure(bg="#7571E6")
window.resizable(False, False)

canvas = Canvas(
    window,
    bg="#7571E6",
    height=308,
    width=387,
    bd=0,
    highlightthickness=0,
    relief="ridge",
)
canvas.place(x=0, y=0)

# PAGES #
def load_main_page():
    create_text("COCONUTZ\nCoconut Type Classifier", 37, 37, 313, 61, 24)

    dc_btn=btn("Data Collection", 77, 131, 233, 46, 14, lambda: switch_page("data_collection1"))
    btn_hover(dc_btn)

    dt_btn=btn("Data Detection", 77, 219, 233, 46, 14, lambda: switch_page("data_detection1"))
    btn_hover(dt_btn)
    
def load_data_collection_page_1():
    global video_capture, picam, camera_after, selected_file
    create_text("Data Collection (Camera)", 37, 10, 313, 40, 20)

    cam_canvas = Canvas(window, width=320, height=200, bg="#5A56C8",
                        highlightthickness=0, bd=0)
    cam_canvas.place(x=33, y=45)

    selected_file = None

    stop_camera()

    # --- camera setup ---
    if USE_PICAMERA2:
        print("Using PiCamera2 (Raspberry Pi Camera)")
        picam = PiCamera()
        preview_config = picam.create_preview_configuration(main={"size": (640, 480)})
        picam.configure(preview_config)
        picam.start()

        def update_frame_picam():
            global camera_after_id
            try:
                frame = picam.capture_array()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (320, 200))
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                # try drawing; if canvas removed, exception will be caught
                cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
                cam_canvas.image = imgtk
            except Exception:
                # canvas likely destroyed or camera stopped; stop loop
                stop_camera()
                return
            camera_after = window.after(10, update_frame_picamera)
        
        def capture_and_next():
            try:
                frame = picam.capture_array()
                timestamp = time.strftime('%Y%m%d-%H%M%S')
                tmp_name = f"tmp_capture_{timestamp}.jpg"
                tmp_path = folder_path1 / tmp_name
                # save as BGR for OpenCV compatibility
                cv2.imwrite(str(tmp_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                selected_file = str(tmp_path)
                stop_camera()
                switch_page("data_collection2")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to Capture: {e}")
        update_frame_picam()

    else:
        print("Using OpenCV VideoCapture (Webcam)")
        global video_capture, camera_after
        video_capture = cv2.VideoCapture(0)

        def update_frame_cam():
            global camera_after
            try:
                if video_capture is None:
                    return
                ret, frame = video_capture.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (320, 200))
                    img = Image.fromarray(frame_resized)
                    imgtk = ImageTk.PhotoImage(image=img)
                    cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
                    cam_canvas.image = imgtk
                else:
                    pass
            except Exception:
                stop_camera()
                return
            camera_after = window.after(10, update_frame_cam)

        def capture_and_next():
            global selected_file
            try:
                if video_capture is None:
                    messagebox.showerror("Camera Error", "Camera not initialized.")
                    return
                ret, frame = video_capture.read()
                if ret:
                    timestamp = time.strftime('%Y%m%d-%H%M%S')
                    tmp_name = f"tmp_capture_{timestamp}.jpg"
                    tmp_path = folder_path1 / tmp_name
                    cv2.imwrite(str(tmp_path), frame)
                    selected_file = str(tmp_path)
                    stop_camera()
                    switch_page("data_collection2")
                else:
                    messagebox.showerror("Capture Error", "Failed to read frame from camera.")
            except Exception as e:
                messagebox.showerror("Capture Error", f"Failed to Capture: {e}")

        update_frame_cam()

    capture_btn=btn("Feature Extraction", 110, 260, 160, 27, 14, capture_and_next)
    btn_hover(capture_btn)

def load_data_collection_page_2():
    create_text("What is the classification of the\nCoconut?", 37, 19, 313, 50, 22)

    def set_label_and_next(label):
        global selected_label
        selected_label = label
        switch_page("data_collection3")

    malakanin_btn=btn("Malakanin", 77, 92, 233, 46, 14, lambda: set_label_and_next("malakanin"))
    btn_hover(malakanin_btn)

    malauhog_btn=btn("Malauhog", 77, 163, 233, 46, 14, lambda: set_label_and_next("malauhog"))
    btn_hover(malauhog_btn)

    malakatad_btn=btn("Malakatad", 77, 234, 233, 46, 14, lambda: set_label_and_next("malakatad"))
    btn_hover(malakatad_btn)

def load_data_collection_page_3():
    create_text("Are you sure this is\nthe right classification?", 35, 50, 321, 96, 30)

    def confirm_yes():
        global selected_file, selected_label
        if selected_file and selected_label:        
            features = camera_features(selected_file)  

        save_features_to_csv(features, csv_path, selected_label)

        try:
            src = Path(selected_file)
            # final filename: <label>_YYYYMMDD-HHMMSS.jpg
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            dst_name = f"{selected_label}_{timestamp}{src.suffix}"
            dst_path = folder_path1 / dst_name
            src.replace(dst_path)  
            selected_file = str(dst_path)
        except Exception as e:
            print("Warning: could not rename/move image:", e)

        camera_prepro(selected_file)

        switch_page("data_collection4")


    yes=btn("Yes", 53, 173, 121, 53, 14, confirm_yes)
    btn_hover(yes)
    no=btn("No", 206, 173, 121, 53, 14, lambda: switch_page("data_collection2"))
    btn_hover(no)

def load_data_collection_page_4():
    create_text("Would you like to\ncollect more data?", 33, 50, 321, 96, 30)

    yes=btn("Yes", 62, 169, 121, 53, 14, lambda: switch_page("data_collection1"))
    btn_hover(yes)

    mm_btn=btn("Main Menu", 218, 169, 121, 53, 14, lambda: switch_page("main"))
    btn_hover(mm_btn)

# --- CAMERA PAGE INTEGRATED HERE ---
def load_data_detection_page_1():
    global video_capture, picam, camera_after, selected_file
    create_text("Image Capture", 37, 10, 313, 40, 20)
    cam_canvas = Canvas(window, width=320, height=200, bg="#5A56C8",
                        highlightthickness=0, bd=0)
    cam_canvas.place(x=33, y=45)

    # ensure selected_file cleared on entry
    selected_file = None

    stop_camera()

    # --- camera setup ---
    if USE_PICAMERA2:
        print("Using PiCamera2 (Raspberry Pi Camera)")
        picam = PiCamera()
        preview_config = picam.create_preview_configuration(main={"size": (640, 480)})
        picam.configure(preview_config)
        picam.start()

        def update_frame_picam():
            global camera_after_id
            try:
                frame = picam.capture_array()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (320, 200))
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                # try drawing; if canvas removed, exception will be caught
                cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
                cam_canvas.image = imgtk
            except Exception:
                # canvas likely destroyed or camera stopped; stop loop
                stop_camera()
                return
            camera_after = window.after(10, update_frame_picamera)
        
        def capture_and_next():
            try:
                frame = picam.capture_array()
                timestamp = time.strftime('%Y%m%d-%H%M%S')
                tmp_name = f"tmp_capture_{timestamp}.jpg"
                tmp_path = folder_path1 / tmp_name
                # save as BGR for OpenCV compatibility
                cv2.imwrite(str(tmp_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                selected_file = str(tmp_path)
                stop_camera()
                switch_page("data_collection2")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to Capture: {e}")
        update_frame_picam()

    else:
        print("Using OpenCV VideoCapture (Webcam)")
        global video_capture, camera_after
        video_capture = cv2.VideoCapture(0)

        def update_frame_cam():
            global camera_after
            try:
                if video_capture is None:
                    return
                ret, frame = video_capture.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (320, 200))
                    img = Image.fromarray(frame_resized)
                    imgtk = ImageTk.PhotoImage(image=img)
                    cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
                    cam_canvas.image = imgtk
                else:
                    pass
            except Exception:
                stop_camera()
                return
            camera_after = window.after(10, update_frame_cam)

        def capture_and_next():
            global selected_file
            try:
                if video_capture is None:
                    messagebox.showerror("Camera Error", "Camera not initialized.")
                    return
                ret, frame = video_capture.read()
                if ret:
                    timestamp = time.strftime('%Y%m%d-%H%M%S')
                    tmp_name = f"tmp_capture_{timestamp}.jpg"
                    tmp_path = folder_path1 / tmp_name
                    cv2.imwrite(str(tmp_path), frame)
                    selected_file = str(tmp_path)
                    stop_camera()
                    switch_page("data_detection2")
                else:
                    messagebox.showerror("Capture Error", "Failed to read frame from camera.")
            except Exception as e:
                messagebox.showerror("Capture Error", f"Failed to Capture: {e}")

        update_frame_cam()

    capture_btn=btn("Feature Extraction", 110, 260, 160, 27, 14, capture_and_next)
    btn_hover(capture_btn)

def load_data_detection_page_2():
    global selected_file

    create_text("Captured Image", 37, 10, 313, 40, 20)

    if not selected_file or not os.path.exists(selected_file):
        create_text("No image captured!", 37, 60, 313, 200, 16)
    
    try: 
        img=Image.open(selected_file) 
        img= img.resize((320, 200), Image.LANCZOS)
        imgtk=ImageTk.PhotoImage(image=img)

        canvas=Canvas(window, width=320, height=200, bg="#5A56C8",
                        highlightthickness=0, bd=0)
        canvas.place(x=33, y=45)
        canvas.create_image(0, 0, image=imgtk, anchor="nw")
        canvas.image=imgtk

    except Exception as e:
        create_text("Error loading image!", 37, 60, 313, 200, 16)

    again=btn("Try Again", 70, 270, 100, 26, 14, lambda: switch_page("data_detection1"))
    btn_hover(again)

    def on_proceed():
        if selected_file and os.path.exists(selected_file):
            camera_prepro(selected_file)
            switch_page("data_detection3")

    next=btn("Proceed", 220, 270, 100, 26, 14, lambda: switch_page("data_detection3"))
    btn_hover(next)

def load_data_detection_page_3(): # So
    

    def camera_class():
        global hsv_class
        if hsv_class is not None:
            switch_page("data_detection4")

        else:
            window.after(500, camera_class)
    
    def processing():
        if selected_file:
            camera_prepro(selected_file)
            camera_class()

    create_text("Processing Image...", 50, 100, 300, 100, 16)
    processing()



# SOLENOID AND AUDIO #
def load_data_detection_page_4():
    create_text("Audio Data Capture", 33, 80, 321, 100, 20)

    next=btn("Activate Solenoid", 115, 220, 154, 53, lambda: switch_page("data_detection5"))
    btn_hover(next)


def load_data_detection_page_5():
    global selected_file, hsv_class

    # Placeholder for Data for Audio #
    audio_class = {
        "malakanin": 0.0,
        "malauhog": 0.0,
        "malakatad": 0.0
    }

    create_text("Fuzzy Logic Summary", 37, 10, 313, 40, 20)

    if selected_file and os.path.exists(selected_file):
        try:
            img = Image.open(selected_file)
            img = img.resize((120, 90), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas_img = Canvas(window, width=120, height=90, bg="#5A56C8",
                                highlightthickness=0, bd=0)
            canvas_img.place(x=30, y=70)
            canvas_img.create_image(0, 0, image=imgtk, anchor="nw")
            canvas_img.image = imgtk
        except Exception as e:
            print("Error loading image:", e)
    else:
        create_text("No image available.", 37, 80, 313, 40, 14)

    if hsv_class:
        final_class = max(hsv_class, key=hsv_class.get)
    else:
        final_class = "N/A"

    create_text(f"Final Class: {final_class}", 37, 170, 313, 30, 18)

    create_text("Camera Classification", 50, 210, 160, 25, 13)
    create_text("Audio Classification", 220, 210, 160, 25, 13)

    y_start = 235
    spacing = 25

    classes = ["malakanin", "malauhog", "malakatad"]
    for i, cls in enumerate(classes):
        cam_val = hsv_class.get(cls, 0.0) if hsv_class else 0.0
        aud_val = audio_class.get(cls, 0.0)
        create_text(f"{cls}: {cam_val:.2f}", 50, y_start + i * spacing, 160, 20, 12)
        create_text(f"{cls}: {aud_val:.2f}", 220, y_start + i * spacing, 160, 20, 12)

    data_cap_dir = Path("Data Detection Captures")
    data_cap_dir.mkdir(exist_ok=True)
    save_path = data_cap_dir / f"{Path(selected_file).stem}_{final_class}.jpg"
    try:
        import shutil
        shutil.copy(selected_file, save_path)
        print(f"[INFO] Saved classified image as: {save_path}")
    except Exception as e:
        print(f"[WARN] Failed to save classified image: {e}")


    back = btn("Back to Main", 250, 270, 150, 26, 14, lambda: switch_page("main"))
    btn_hover(back)


# APP START #
switch_page("main")
window.mainloop()