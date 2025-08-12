import tkinter as tk
from tkinter import filedialog
import imageio
import numpy as np
from PIL import Image, ImageTk
import imageio.v2 as imageio
from logisticRegression import receiveImageInfo

def load_image():
    global img_np
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")])
    if file_path:
        img = imageio.imread(file_path)
        img_np = np.array(img)  
        display_image()
        text = receiveImageInfo(img_np)
        show_text_in_window(text)

def display_image():
    if img_np is not None:
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.resize((400, 300))
        img_tk = ImageTk.PhotoImage(img_pil)
        
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.image = img_tk
        
        result_label.config(text="Image displayed below:")


def show_text_in_window(text):
    result_label.config(text=text)
        
root = tk.Tk()
root.title("Cat Classfier")

window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")


load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()


canvas = tk.Canvas(root, width=400, height=300)
canvas.pack()


result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()


