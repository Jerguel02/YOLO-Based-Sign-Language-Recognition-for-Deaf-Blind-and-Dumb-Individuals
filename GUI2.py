import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QEventLoop, QTimer, Qt, QUrl, pyqtSignal
from PyQt5.QtMultimedia import QSound
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import wave
import cv2
from ultralytics import YOLO
from PIL import ImageQt
from datetime import datetime
import pyttsx3



class YOLO_GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO-Based Object Detection GUI")
        self.setGeometry(100, 100, 800, 600)

        #Vid Label
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        #Gestute
        self.message_label = QLabel("Message", self)
        self.message_label.setAlignment(Qt.AlignCenter)

        #Text box (Gesture)
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setMaximumHeight(30)  
        
        #Emotion
        self.emotion_label = QLabel("Emotion", self)
        self.emotion_label.setAlignment(Qt.AlignCenter)

        #Text box(Emotion)
        self.emotion_display = QTextEdit(self)
        self.emotion_display.setReadOnly(True)
        self.emotion_display.setMaximumHeight(30)

        #Submit
        self.submit_button = QPushButton("Submit", self)
        self.submit_button.setMaximumHeight(30) 
        self.submit_button.clicked.connect(self.submit_chat)

        chat_layout = QHBoxLayout()
        chat_layout.addWidget(self.message_label)
        chat_layout.addWidget(self.chat_display)
        emotion_layout = QHBoxLayout()
        emotion_layout.addWidget(self.emotion_label)
        emotion_layout.addWidget(self.emotion_display)
        emotion_layout.addWidget(self.submit_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addLayout(chat_layout)
        main_layout.addLayout(emotion_layout)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

     
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms

        self.video_source = 0  # Camera
        self.video = cv2.VideoCapture(self.video_source)

        self.list_of_emotion = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.list_of_gesture = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "additional", "alcohol", "allergy", "bacon", "bag", "barbecue", "bill", "biscuit", "bitter", "bread", "burger", "bye", "cake", "cash", "cheese", "chicken", "coke", "cold", "cost", "coupon", "credit card", "cup", "dessert", "drink", "drive", "eat", "eggs", "enjoy", "fork", "french fries", "fresh", "hello", "hot", "icecream", "ingredients", "juicy", "ketchup", "lactose", "lettuce", "lid", "manager", "menu", "milk", "mustard", "napkin", "no", "order", "pepper", "pickle", "pizza", "please", "ready", "receipt", "refill", "repeat", "safe", "salt", "sandwich", "sauce", "small", "soda", "sorry", "spicy", "spoon", "straw", "sugar", "sweet", "thank-you", "tissues", "tomato", "total", "urgent", "vegetables", "wait", "warm", "water", "what", "would", "yoghurt", "your"]
        self.pre_name_of_emotion = ""
        self.pre_name_of_gesture = ""
        self.model = YOLO("YOLOv8Checkpoint/YOLOv8Checkpoint/train4/weights/best.pt")

        self.last_detected_time = None
        self.chat_text = ""
        self.emotion_text = ""
        self.engine = pyttsx3.init()
        self.chat_beep = QSound("beep.wav")
        self.emotion_beep = QSound("double-beep.wav")
    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            results = self.model.predict(frame, show=False)
            if results and len(results[0].boxes) > 0:
                names = self.model.names

                for i, det in enumerate(results[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, det[:4])
                    cls = int(results[0].boxes.cls[i])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, names[cls], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if len(names[cls]) > 0 :
                        #current_detected_time = datetime.now()
                        if self.pre_name_of_gesture != names[cls]:
                            #if self.last_detected_time is not None:
                            #time_difference = (current_detected_time - self.last_detected_time).total_seconds()
                            objects = "".join(names[int(cls)])
                            #if time_difference >= 1:
                            for c in self.list_of_gesture:
                                if c == names[int(cls)]:
                                    self.chat_text += f"{objects} "
                                    self.chat_display.insertPlainText(f"{objects} ")
                                    self.pre_name_of_gesture = names[cls]
                                    break
                        if self.pre_name_of_emotion != names[cls]:
                            objects = "".join(names[int(cls)])
                            for c in self.list_of_emotion:
                                if c == names[int(cls)]:
                                    self.emotion_display.clear()
                                    self.emotion_text = f"{objects}"
                                    self.emotion_display.insertPlainText(f"{objects}") 
                                    self.pre_name_of_emotion = names[cls]
                                    break
                            #self.last_detected_time = current_detected_time
                                    
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytesPerLine = ch * w
            qImg = QImage(frame_rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)
            qImg = qImg.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)

            self.label.setPixmap(QPixmap.fromImage(qImg))

    def speak_chat(self):
        print("Message: " + self.chat_text)
        self.speak(self.chat_text, 150)


        
    def speak_emotion(self):   
        print("The person you are talking to seems to be: " + self.emotion_text)
        self.speak("The person you are talking to seems to be: " + self.emotion_text, 150)


    def speak(self, text, rate):
        
        self.engine.setProperty('rate', rate)
        self.engine.say(text)
        self.engine.runAndWait() 

    def submit_chat(self):
        self.timer.stop()

        self.chat_beep.play()
        self.speak_chat()
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_()
        self.speak_emotion()
        self.emotion_beep.play()
        self.chat_text = ""
        self.emotion_text = ""
        self.chat_display.clear()
        self.emotion_display.clear()
        self.timer.start(30)

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = YOLO_GUI()
    mainWindow.show()
    sys.exit(app.exec_())
