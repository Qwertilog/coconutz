from pathlib import Path
from tkinter import Tk, Canvas, Text, Button, PhotoImage, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
import numpy as np
import pandas as pd
import sys


def get_base_path():
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / 'Draft1' / 'assets'
    return Path(__file__).parent / 'Draft1' / 'assets'

ASSETS_PATH = get_base_path()


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


selected_file = None
selected_label = None

if getattr(sys, 'frozen', False):  # Running as EXE
    base_dir = Path(sys.executable).parent
else:  # Running as script
    base_dir = Path(__file__).parent

csv_path = base_dir / "coconut_features.csv"


def switch_page(page_name):
    for widget in window.winfo_children():
        if widget is not canvas:
            widget.destroy()

    pages = {
        "main": load_main_page,
        "data_collection1": load_data_collection_page_1,
        "data_collection2": load_data_collection_page_2,
        "data_collection3": load_data_collection_page_3,
        "data_collection4": load_data_collection_page_4,
    }

    if page_name in pages:
        pages[page_name]()

# UI HELP #
def add_hover_effect(button, img_normal, img_hover):
    def on_enter(e): button.config(image=img_hover)
    def on_leave(e): button.config(image=img_normal)
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

def create_button(img_normal, img_hover, x, y, w, h, command):
    btn = Button(
        image=img_normal,
        borderwidth=0,
        highlightthickness=0,
        command=command,
        relief="flat",
    )
    btn.place(x=x, y=y, width=w, height=h)
    add_hover_effect(btn, img_normal, img_hover)
    return btn


def open_file_dialog():
    global selected_file
    filepath = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if filepath:
        selected_file = filepath
        drop_area.config(text=f"Selected:\n{Path(filepath).name}")

def on_file_drop(event):
    global selected_file
    filepath = event.data.strip("{}")
    selected_file = filepath
    drop_area.config(text=f"Dropped:\n{Path(filepath).name}")

# HSV FEATURE EXTRACTION #
def extract_hsv_features(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Di pa adjusted 
    lower = np.array([20, 40, 40])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    masked_hsv = hsv[mask > 0]
    h_mean = np.mean(masked_hsv[:, 0]) if masked_hsv.size > 0 else 0
    s_mean = np.mean(masked_hsv[:, 1]) if masked_hsv.size > 0 else 0
    v_mean = np.mean(masked_hsv[:, 2]) if masked_hsv.size > 0 else 0

    return [h_mean, s_mean, v_mean]

def save_features_to_csv(features, filepath, label):
    df = pd.DataFrame(
        [[label] + features],
        columns=["Label", "H_mean", "S_mean", "V_mean"]
    )
    df.to_csv(filepath, mode="a", header=not Path(filepath).exists(), index=False)

# MAIN WINDOW #
window = TkinterDnD.Tk()
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
    create_text("COCOKNUTZ\nCoconut Type Classifier", 37, 37, 313, 61, 24)

    datacollection_img = PhotoImage(file=relative_to_assets("datacollection_1.png"))
    datacollection_hover = PhotoImage(file=relative_to_assets("datacollection_2.png"))
    create_button(
        datacollection_img, datacollection_hover,
        77, 131, 233, 46,
        lambda: switch_page("data_collection1")
    )

    datadetection_img = PhotoImage(file=relative_to_assets("datadetection_1.png"))
    datadetection_hover = PhotoImage(file=relative_to_assets("datadetection_2.png"))
    create_button(
        datadetection_img, datadetection_hover,
        77, 219, 233, 46,
        lambda: switch_page("main")
    )

def load_data_collection_page_1():
    global drop_area
    drop_area = Button(
        window,
        text="Drag & Drop Image Here\nor Click to Browse",
        bg="#5A56C8", fg="white",
        relief="ridge", bd=2
    )
    drop_area.place(x=60, y=80, width=260, height=120)
    drop_area.config(command=open_file_dialog)

    drop_area.drop_target_register(DND_FILES)
    drop_area.dnd_bind("<<Drop>>", on_file_drop)

    next_img = PhotoImage(file=relative_to_assets("extraction_1.png"))
    next_hover = PhotoImage(file=relative_to_assets("extraction_2.png"))
    create_button(next_img, next_hover, 268, 262, 95, 18, lambda: switch_page("data_collection2"))

def load_data_collection_page_2():
    create_text("What is the classification of the\nCoconut?", 37, 19, 313, 50, 22)

    def set_label_and_next(label):
        global selected_label
        selected_label = label
        switch_page("data_collection3")

    malakanin_img = PhotoImage(file=relative_to_assets("malakanin_1.png"))
    malakanin_hover = PhotoImage(file=relative_to_assets("malakanin_2.png"))
    create_button(
        malakanin_img, malakanin_hover,
        77, 92, 233, 46,
        lambda: set_label_and_next("malakanin")
    )

    malauhog_img = PhotoImage(file=relative_to_assets("malauhog_1.png"))
    malauhog_hover = PhotoImage(file=relative_to_assets("malauhog_2.png"))
    create_button(
        malauhog_img, malauhog_hover,
        77, 163, 233, 46,
        lambda: set_label_and_next("malauhog")
    )

    malakatad_img = PhotoImage(file=relative_to_assets("malakatad_1.png"))
    malakatad_hover = PhotoImage(file=relative_to_assets("malakatad_2.png"))
    create_button(
        malakatad_img, malakatad_hover,
        77, 234, 233, 46,
        lambda: set_label_and_next("malakatad")
    )

def load_data_collection_page_3():
    create_text("Are you sure this is\nthe right classification?", 35, 50, 321, 96, 30)

    yes_img = PhotoImage(file=relative_to_assets("yes_1.png"))
    yes_hover = PhotoImage(file=relative_to_assets("yes_2.png"))

    def confirm_yes():
        if selected_file and selected_label:
            features = extract_hsv_features(selected_file)
            save_features_to_csv(features, csv_path, selected_label)
        switch_page("data_collection4")

    create_button(yes_img, yes_hover, 53, 173, 121, 53, confirm_yes)

    no_img = PhotoImage(file=relative_to_assets("no_1.png"))
    no_hover = PhotoImage(file=relative_to_assets("no_2.png"))
    create_button(no_img, no_hover, 206, 173, 121, 53, lambda: switch_page("data_collection2"))

def load_data_collection_page_4():
    create_text("Would you like to\ncollect more data?", 33, 50, 321, 96, 30)

    yes_img = PhotoImage(file=relative_to_assets("yes_1.png"))
    yes_hover = PhotoImage(file=relative_to_assets("yes_2.png"))
    create_button(
        yes_img, yes_hover,
        62, 169, 121, 53,
        lambda: switch_page("data_collection1")
    )

    menu_img = PhotoImage(file=relative_to_assets("menu_1.png"))
    menu_hover = PhotoImage(file=relative_to_assets("menu_2.png"))
    create_button(
        menu_img, menu_hover,
        218, 169, 121, 53,
        lambda: switch_page("main")
    )

def load_data_detection_page_1():
    create_text("Data Detection Page\n(Placeholder)", 37, 100, 313, 61, 24)

# APP START #
switch_page("main")
window.mainloop()
