import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from surveillance_engine import SurveillanceEngine

class SurveillanceDashboard:

    def __init__(self, root):

        self.root = root
        self.root.title("AI Smart Surveillance System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e1e")

        self.engine = SurveillanceEngine()
        self.cap = None
        self.running = False

        # ===== Title =====
        title = tk.Label(
            root,
            text="AI SMART SURVEILLANCE SYSTEM",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#1e1e1e"
        )
        title.pack(pady=10)

        # ===== Main Frame =====
        main_frame = tk.Frame(root, bg="#1e1e1e")
        main_frame.pack()

        # ===== Video Frame =====
        video_frame = tk.Frame(main_frame, bg="black", width=700, height=500)
        video_frame.grid(row=0, column=0, padx=10)

        self.video_label = tk.Label(video_frame)
        self.video_label.pack()

        # ===== Side Panel =====
        side_panel = tk.Frame(main_frame, bg="#2a2a2a", width=250)
        side_panel.grid(row=0, column=1, sticky="ns")

        # Status
        self.status_label = tk.Label(
            side_panel,
            text="Status: Idle",
            fg="white",
            bg="#2a2a2a",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=10)

        # Buttons
        tk.Button(
            side_panel,
            text="Start Camera",
            width=20,
            command=self.start_camera
        ).pack(pady=5)

        tk.Button(
            side_panel,
            text="Load Video",
            width=20,
            command=self.load_video
        ).pack(pady=5)

        tk.Button(
            side_panel,
            text="Stop",
            width=20,
            command=self.stop
        ).pack(pady=5)

    # ===== Start Camera =====
    def start_camera(self):

        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.status_label.config(text="Status: Camera Running")

        self.update_frame()

    # ===== Load Video =====
    def load_video(self):

        path = filedialog.askopenfilename()

        if path:
            self.cap = cv2.VideoCapture(path)
            self.running = True
            self.status_label.config(text="Status: Video Running")

            self.update_frame()

    # ===== Stop =====
    def stop(self):

        self.running = False

        if self.cap:
            self.cap.release()

        self.status_label.config(text="Status: Stopped")

    # ===== Update Video Frame =====
    def update_frame(self):

        if self.running and self.cap:

            ret, frame = self.cap.read()

            if ret:

                frame= self.engine.process_frame(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(frame)

                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(40, self.update_frame)


# ===== Run GUI =====
root = tk.Tk()

app = SurveillanceDashboard(root)

root.mainloop()