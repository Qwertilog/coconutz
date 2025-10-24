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
import subprocess
from joblib import load
from picamera2 import PiCamera2
from PIL import Image, ImageTk

video_capture = None
rpicam_proc = None
camera_after = None

selected_file = None
selected_label = None
processed_file = None
hsv_class = None

USE_RPICAM = False

# --- Try to detect RPiCam stack availability ---
try:
    from picamera2 import Picamera2
    USE_RPICAM = True
except ImportError:
    USE_RPICAM = False


# --- Load model if available ---
try:
    model = load("camera_model.pkl")
except Exception as e:
    print(f"[WARN] Model not loaded: {e}")
    model = None


def stop_camera():
    global video_capture, rpicam_proc, camera_after
    try:
        if camera_after is not None:
            window.after_cancel(camera_after)
    except Exception:
        pass
    camera_after = None

    try:
        if video_capture is not None and hasattr(video_capture, "isOpened") and video_capture.isOpened():
            video_capture.release()
        video_capture = None
    except Exception:
        video_capture = None

    try:
        if rpicam_proc is not None:
            rpicam_proc.terminate()
            rpicam_proc = None
    except Exception:
        rpicam_proc = None


if getattr(sys, 'frozen', False):  # Running as EXE
    base_dir = Path(sys.executable).parent
else:
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
    b = Button(
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
    b.place(x=x, y=y, width=w, height=h)
    return b


def btn_hover(button, normal="#C82333", hover="#E74C3C"):
    def on_enter(e): button.config(bg=hover)
    def on_leave(e): button.config(bg=normal)
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)


def create_text(content, x, y, w, h, font_size, style):
    txt = Text(
        window,
        bd=0,
        bg="#7571E6",
        fg="#FFFFFF",
        highlightthickness=0,
        font=("InriaSans Regular", font_size, style),
    )
    txt.insert("1.0", content)
    txt.tag_configure("center", justify="center")
    txt.tag_add("center", "1.0", "end")
    txt.place(x=x, y=y, width=w, height=h)
    return txt


# --- Camera Processing --- #
def get_rpicam_frame():
    """Capture a frame using the PiCamera2 interface."""
    global picam2
    try:
        if 'picam2' not in globals() or picam2 is None:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            time.sleep(0.2)  # allow sensor to warm up
        frame = picam2.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[ERROR] PiCamera2 frame capture failed: {e}")
        return None


def camera_prepro(image_path):
    def task():
        global processed_file, hsv_class
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Unreadable image")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([5, 30, 30])
            upper = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            resized_img = cv2.resize(masked_img, (224, 224))
            processed_file = os.path.splitext(image_path)[0] + "_processed.jpg"
            cv2.imwrite(processed_file, resized_img)
            features = camera_features(processed_file)
            if model is not None:
                x = np.array([features])
                probs = model.predict_proba(x)[0]
                class_labels = model.classes_
                hsv_class = {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}
            else:
                print("[INFO] No model — only preprocessing done.")
        except Exception as e:
            print(f"[ERROR] Preprocessing/classification failed: {e}")

    threading.Thread(target=task).start()


def camera_features(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    return [h_mean, s_mean, v_mean]


def save_features_to_csv(features, filepath, label):
    df = pd.DataFrame([[label] + features], columns=["Label", "H_mean", "S_mean", "V_mean"])
    df.to_csv(filepath, mode="a", header=not Path(filepath).exists(), index=False)


# MAIN WINDOW #
window = Tk()
window.geometry("800x420")
window.configure(bg="#7571E6")
window.resizable(False, False)

canvas = Canvas(window, bg="#7571E6", height=308, width=387, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)


# --- MAIN PAGE --- #
def load_main_page():
    create_text("COCONUTZ\nCoconut Type Classifier", 45, 60, 700, 150, 45, "bold")
    dc_btn = btn("Data Collection", 180, 240, 450, 46, 18, lambda: switch_page("data_collection1"))
    btn_hover(dc_btn)
    dt_btn = btn("Data Detection", 180, 315, 450, 46, 18, lambda: switch_page("data_detection1"))
    btn_hover(dt_btn)


# --- CAMERA PAGE FIX --- #
def camera_stream(cam_canvas, folder_path, next_page):
    global camera_after

    def update():
        global camera_after
        frame = get_rpicam_frame() if USE_RPICAM else (lambda cap=video_capture: cap.read()[1] if cap.isOpened() else None)()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (560, 280))
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
            cam_canvas.image = imgtk
        camera_after = window.after(100, update)

    def capture_and_next():
        global selected_file
        frame = get_rpicam_frame() if USE_RPICAM else (lambda cap=video_capture: cap.read()[1] if cap.isOpened() else None)()
        if frame is not None:
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            tmp_name = f"tmp_capture_{timestamp}.jpg"
            tmp_path = folder_path / tmp_name
            cv2.imwrite(str(tmp_path), frame)
            selected_file = str(tmp_path)
            stop_camera()
            switch_page(next_page)
        else:
            messagebox.showerror("Capture Error", "Failed to capture frame.")

    update()
    return capture_and_next


# --- Data Collection Page 1 --- #
def load_data_collection_page_1():
    global video_capture
    create_text("Data Collection (Camera)", 195, 20, 400, 300, 24, "bold")
    cam_canvas = Canvas(window, width=560, height=280, bg="#5A56C8", highlightthickness=0, bd=0)
    cam_canvas.place(x=120, y=70)
    stop_camera()
    if not USE_RPICAM:
        video_capture = cv2.VideoCapture(0)
    capture_btn = btn("Feature Extraction", 280, 360, 215, 35, 18,
                      camera_stream(cam_canvas, folder_path1, "data_collection2"))
    btn_hover(capture_btn)

def load_data_collection_page_2():
    create_text("What is the classification of the\nCoconut?", 155, 50, 500, 400, 24, "bold")

    def set_label_and_next(label):
        global selected_label
        selected_label = label
        switch_page("data_collection3")

    malakanin_btn=btn("Malakanin", 200, 160, 400, 46, 20, lambda: set_label_and_next("malakanin"))
    btn_hover(malakanin_btn)

    malauhog_btn=btn("Malauhog", 200, 240, 400, 46, 20, lambda: set_label_and_next("malauhog"))
    btn_hover(malauhog_btn)

    malakatad_btn=btn("Malakatad", 200, 320, 400, 46, 20, lambda: set_label_and_next("malakatad"))
    btn_hover(malakatad_btn)

def load_data_collection_page_3():
    create_text("Are you sure this is\nthe right classification?", 75, 75, 650, 400, 42, "bold")

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


    yes=btn("Yes", 225, 250, 150, 80, 18, confirm_yes)
    btn_hover(yes)
    no=btn("No", 450, 250, 150, 80, 18, lambda: switch_page("data_collection2"))
    btn_hover(no)

def load_data_collection_page_4():
    create_text("Would you like to\ncollect more data?", 75, 50, 640, 400, 48, "bold")

    yes=btn("Yes", 215, 250, 150, 80, 18, lambda: switch_page("data_collection1"))
    btn_hover(yes)

    mm_btn=btn("Main Menu", 435, 250, 150, 80, 18, lambda: switch_page("main"))
    btn_hover(mm_btn)

def load_data_detection_page_1():
    global video_capture
    create_text("Image Capture", 200, 15, 400, 300, 32, "bold")
    cam_canvas = Canvas(window, width=560, height=280, bg="#5A56C8",
                        highlightthickness=0, bd=0)
    cam_canvas.place(x=120, y=70)

    if not USE_RPICAM:
        video_capture = cv2.VideoCapture(0)
    capture_btn = btn("Feature Extraction", 280, 360, 215, 35, 18,
                      camera_stream(cam_canvas, folder_path1, "data_collection2"))
    btn_hover(capture_btn)

def load_data_detection_page_2():
    global selected_file

    create_text("Captured Image", 200, 15, 400, 300, 32, "bold")

    if not selected_file or not os.path.exists(selected_file):
        create_text("No image captured!", 37, 60, 313, 200, 16)
    
    try: 
        img=Image.open(selected_file) 
        img= img.resize((560, 280), Image.LANCZOS)
        imgtk=ImageTk.PhotoImage(image=img)

        cam_canvas = Canvas(window, width=560, height=280, bg="#5A56C8",
                        highlightthickness=0, bd=0)
        cam_canvas.place(x=120, y=70)
        cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
        cam_canvas.image=imgtk

    except Exception as e:
        create_text("Error loading image!", 37, 60, 313, 200, 16)

    again=btn("Try Again", 160, 360, 215, 35, 18, lambda: switch_page("data_detection1"))
    btn_hover(again)

    def on_proceed():
        if selected_file and os.path.exists(selected_file):
            camera_prepro(selected_file)
            switch_page("data_detection3")

    next=btn("Proceed", 425, 360, 215, 35, 18, lambda: switch_page("data_detection3"))
    btn_hover(next)

def load_data_detection_page_3():
    

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

    create_text("Processing Image...", 250, 200, 300, 100, 18, "normal")
    processing()



def load_data_detection_page_4():
    create_text("Audio Data Capture", 50, 125, 700, 100, 48, "bold")

    next=btn("Activate Solenoid", 290, 220, 210, 65, 18, lambda: switch_page("data_detection5"))
    btn_hover(next)


def load_data_detection_page_5():
    global selected_file, hsv_class

    audio_class = {
        "malakanin": 0.0,
        "malauhog": 0.0,
        "malakatad": 0.0
    }

    create_text("Classification Summary", 160, 30, 500, 40, 24, "bold")

    if selected_file and os.path.exists(selected_file):
        try:
            img = Image.open(selected_file)
            img = img.resize((320, 240), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas_img = Canvas(window, width=320, height=240, bg="#5A56C8",
                                highlightthickness=0, bd=0)
            canvas_img.place(x=50, y=100)
            canvas_img.create_image(0, 0, image=imgtk, anchor="nw")
            canvas_img.image = imgtk
        except Exception as e:
            print("Error loading image:", e)
    else:
        create_text("No image available.", 37, 80, 313, 40, 14, "normal")

    if hsv_class:
        final_class = max(hsv_class, key=hsv_class.get)
    else:
        final_class = "N/A"

    create_text(f"Final Class: {final_class}", 410, 120, 313, 30, 18, "bold")

    create_text("Camera Classification", 370, 180, 210, 200, 14, "bold")
    create_text("Audio Classification", 580, 180, 200, 200, 14, "bold")

    y_start = 205
    spacing = 25

    classes = ["malakanin", "malauhog", "malakatad"]
    for i, cls in enumerate(classes):
        cam_val = hsv_class.get(cls, 0.0) if hsv_class else 0.0
        aud_val = audio_class.get(cls, 0.0)
        create_text(f"{cls}: {cam_val:.2f}", 410, y_start + i * spacing, 160, 20, 12, "normal")
        create_text(f"{cls}: {aud_val:.2f}", 600, y_start + i * spacing, 160, 20, 12, "normal")

    data_cap_dir = Path("Data Detection Captures")
    data_cap_dir.mkdir(exist_ok=True)
    save_path = data_cap_dir / f"{Path(selected_file).stem}_{final_class}.jpg"
    try:
        import shutil
        shutil.copy(selected_file, save_path)
        print(f"[INFO] Saved classified image as: {save_path}")
    except Exception as e:
        print(f"[WARN] Failed to save classified image: {e}")


    back = btn("Back to Main", 620, 345, 150, 65, 18, lambda: switch_page("main"))
    btn_hover(back)


# APP START #
switch_page("main")
window.mainloop()
#test#