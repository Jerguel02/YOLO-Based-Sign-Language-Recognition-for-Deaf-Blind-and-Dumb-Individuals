import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import sys

# Hàm nhận diện đối tượng bằng YOLO
def detect_objects(frame):
    # Load model YOLO
    model = YOLO("YOLOv8Checkpoint/YOLOv8Checkpoint/train4/weights/last.pt")
    
    # Dự đoán bằng model
    results = model.predict(frame, show=False)

    # Kiểm tra nếu không có kết quả nào được trả về
    if not results:
        return frame

    # Lặp qua từng kết quả nhận diện và vẽ bounding box lên frame
    for res in results[0]:
        x1, y1, x2, y2, conf, cls = res
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"{int(cls)}: {conf:.2f}", (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO-based Gesture Language and Emotion Recognition ")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Cập nhật frame mỗi 30ms

        self.video = cv2.VideoCapture(0)  # Sử dụng camera mặc định

    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            # Chuyển frame từ OpenCV sang QImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            qImg = qImg.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)

            # Hiển thị frame gốc trên QLabel
            self.label.setPixmap(QPixmap.fromImage(qImg))

            # Nhận diện đối tượng và hiển thị kết quả
            detected_image = detect_objects(frame)
            detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
            h, w, ch = detected_image.shape
            bytesPerLine = ch * w
            qImg = QImage(detected_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
            qImg = qImg.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)

            # Hiển thị frame đã nhận diện trên QLabel
            self.label.setPixmap(QPixmap.fromImage(qImg))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
