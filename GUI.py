from ultralytics import YOLO
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class YOLO_basedReconitionGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0
        self.cap = cv2.VideoCapture(self.video_source)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.canvas = tk.Canvas(window, width = self.width, height = self.height)
        self.canvas.pack()

        self.model = YOLO("YOLOv8Checkpoint/YOLOv8Checkpoint/train4/weights/best.pt")
        self.delay = 10

        
        self.update()
        self.window.mainloop()

    def update(self):

        ret, frame = self.cap.read()
        if ret:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 1
            color = (0,255,0)
            thickness = 2

            #results = self.model.track(frame, persist=True)
            results = self.model.predict(frame, show = False)
            names = self.model.names

            for i, det in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, det)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                #for result in results:
                #    for cls in result.boxes.cls[i]:
                cls = results[0].boxes.cls[i]
                cv2.putText(frame, str(names[int(cls)]), (x1, y1-5), font, fontscale, color, thickness)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            self.window.destroy
        else:
            self.window.after(self.delay, self.update)  # Schedule the next update
        

if __name__ == "__main__":
    YOLO_basedReconitionGUI(tk.Tk(), "YOLO_Based Reconition GUI")


