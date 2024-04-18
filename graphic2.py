import cv2
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

class YOLOGui(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO GUI")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        self.video = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Cập nhật frame mỗi 30ms

        self.model = YOLO("YOLOv8Checkpoint/YOLOv8Checkpoint/train4/weights/last.pt")

    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            qImg = qImg.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(QPixmap.fromImage(qImg))

            # Nhận diện đối tượng
            detected_image = self.detect_objects(frame)
            if detected_image is not None:
                detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
                qImg = QImage(detected_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
                qImg = qImg.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
                self.video_label.setPixmap(QPixmap.fromImage(qImg))

    def detect_objects(self, frame):
        # Dự đoán bằng model
        results = self.model.predict(frame, show=False)

        # Kiểm tra nếu không có kết quả nào được trả về
        if not results or len(results[0]) < 6: # Kiểm tra số lượng phần tử trong results[0]
            return None

        # Lặp qua từng kết quả nhận diện và vẽ bounding box lên frame
        for res in results[0]:
            x1, y1, x2, y2, conf, cls = res
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{int(cls)}: {conf:.2f}", (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOGui()
    window.show()
    sys.exit(app.exec_())
