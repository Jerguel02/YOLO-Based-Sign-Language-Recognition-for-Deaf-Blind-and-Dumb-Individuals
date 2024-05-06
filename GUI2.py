import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QEventLoop, QTimer, Qt, QUrl, pyqtSignal, QThread
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
import pyaudio
import os



class AudioRecorder(QThread):
    signal = pyqtSignal(str)
    #recording_signal = pyqtSignal(bool)
    def __init__(self):
        super().__init__()
        self.audio_filename = "audio/AUDIO.wav"
        self.recording = False
        self.recognizer = sr.Recognizer()
        self.CHUNK = 2048
        self.SAMPLE_FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.FS = 44100
        self.DURATION = 5

    def run(self):
        #recording = True
        p = pyaudio.PyAudio()
        stream = p.open(format=self.SAMPLE_FORMAT,
                        channels=self.CHANNELS,
                        rate=self.FS,
                        frames_per_buffer=self.CHUNK,
                        input=True)
        frames = []
        for _ in range(0, int(self.FS / self.CHUNK * self.DURATION)):
            if self.recording:
                data = stream.read(self.CHUNK)
                frames.append(data)
            else:
                break
        stream.stop_stream()
        stream.close()
        p.terminate()
    
        with wave.open(self.audio_filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.SAMPLE_FORMAT))
            wf.setframerate(self.FS)
            wf.writeframes(b''.join(frames))
    
        recognized_text = self.recognize_speech(self.audio_filename)
        recording = False
        #self.recording_signal.emit(recording)
        self.signal.emit(recognized_text)
    def stop_recording(self):
        print("Thread is terminating...")
        self.recording = False

    def recognize_speech(self, filename):
        with sr.AudioFile(filename) as source:
            audio_data = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio_data)
                print("Recognized text:", text)
                return text
            except (sr.UnknownValueError, sr.RequestError) as e:
                print(f"Speech recognition error: {e}")
                return ""


class YOLO_GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO-Based Object Detection GUI")
        #self.setGeometry(100, 100, 800, 600)
        self.setGeometry(0, 0, QApplication.desktop().screenGeometry().width(), QApplication.desktop().screenGeometry().height())
        self.showFullScreen()
        #Vid Label
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)
        #Gestute
        self.message_label = QLabel("Message", self)
        self.message_label.setAlignment(Qt.AlignLeft)

        #Text box (Gesture)
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        
        self.chat_display.setMaximumHeight(30)  
        #Emotion
        self.emotion_label = QLabel("Emotion", self)
        self.emotion_label.setAlignment(Qt.AlignLeft)

        #Text box(Emotion)
        self.emotion_display = QTextEdit(self)
        self.emotion_display.setReadOnly(True)
        self.emotion_display.setMaximumHeight(30)
        #Submit
        self.submit_button = QPushButton("Submit", self)
        self.submit_button.setMaximumHeight(30) 
        self.submit_button.setMaximumWidth(100)
        self.submit_button.clicked.connect(self.submit_chat)

        

        #--------
        self.sign_language_label = QLabel(self)
        self.sign_language_label.setAlignment(Qt.AlignRight)

        self.recording_label = QLabel("Recording...", self)
        recording_font = QFont()
        recording_font.setPointSize(20)  # Set font size to 16
        self.recording_label.setStyleSheet("color: green;")
        self.recording_label.setFont(recording_font)
        self.recording_label.setAlignment(Qt.AlignCenter)
        self.recording_label.setVisible(False)

        self.stop_recording_label = QLabel("Stopping the recording...", self)
        stop_recording_font = QFont()
        stop_recording_font.setPointSize(20)  # Set font size to 16
        self.stop_recording_label.setStyleSheet("color: red;")
        self.stop_recording_label.setFont(recording_font)
        self.stop_recording_label.setAlignment(Qt.AlignCenter)
        self.stop_recording_label.setVisible(False)

        self.record_button = QPushButton("Record", self)
        self.record_button.setMaximumHeight(30)
        self.record_button.clicked.connect(self.record_audio)

        self.stop_record_button = QPushButton("Stop", self)
        self.stop_record_button.setMaximumHeight(30)
        self.stop_record_button.setVisible(False)
        self.stop_record_button.clicked.connect(self.stop_record)

        self.text_record_label = QLabel("Recored Message", self)
        self.text_record_label.setAlignment(Qt.AlignLeft)

        self.text_record_display = QTextEdit(self)
        self.text_record_display.setReadOnly(True)
        self.text_record_display.setMaximumHeight(50)

        self.text_display = QTextEdit(self)
        self.text_display.setReadOnly(True)
        self.text_display.setMaximumHeight(50)
        #----
        main_layout = QVBoxLayout()
        chat_layout = QHBoxLayout()
        chat_layout.addWidget(self.message_label)
        chat_layout.addWidget(self.chat_display)
        emotion_layout = QHBoxLayout()
        emotion_layout.addWidget(self.emotion_label)
        emotion_layout.addWidget(self.emotion_display)
        emotion_layout.addWidget(self.submit_button)
        
        text_record_layout= QHBoxLayout()
        text_record_layout.addWidget(self.text_record_label)
        text_record_layout.addWidget(self.text_record_display)


        main_layout.addWidget(self.label)
        main_layout.addLayout(chat_layout)
        
        main_layout.addLayout(emotion_layout)
        
        main_layout.addWidget(self.sign_language_label)
        main_layout.addWidget(self.recording_label)
        main_layout.addWidget(self.stop_recording_label)
        main_layout.addWidget(self.record_button)
        main_layout.addWidget(self.stop_record_button)
        main_layout.addLayout(text_record_layout)
        main_layout.addWidget(self.text_record_display)

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

        self.recording_thread = AudioRecorder()
        self.recording_thread.signal.connect(self._recorded_audio_thread)
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
            qImg = qImg.scaled(self.label.width()/2, self.label.height(), Qt.KeepAspectRatio)

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
    #==================================================================
    def record_audio(self):
        self.text_display.clear()
        self.recording_label.setVisible(True)
        self.record_button.setVisible(False)
        self.stop_record_button.setVisible(True)
        self.stop_record_button.setEnabled(True)
        self.recording_thread.recording = True
        self.recording_thread.start()

    def stop_record(self):
        self.stop_record_button.setEnabled(False)
        self.recording_thread.stop_recording()
        self.recording_label.setVisible(False)
        self.stop_recording_label.setVisible(True)
        loop = QEventLoop()
        QTimer.singleShot(3000, loop.quit)
        loop.exec_()
        self.stop_recording_label.setVisible(False)
        self.stop_record_button.setVisible(False)
        self.record_button.setVisible(True)
        

    def _recorded_audio_thread(self, recognized_text):
        if isinstance(recognized_text, str):  
            self.text_record_display.setPlainText(recognized_text)
            self.display_sign_language_image(recognized_text)
            self.stop_record_button.setEnabled(False)
            self.recording_thread.stop_recording()
            self.recording_label.setVisible(False)
            self.stop_recording_label.setVisible(True)
            loop = QEventLoop()
            QTimer.singleShot(3000, loop.quit)
            loop.exec_()
            self.stop_recording_label.setVisible(False)
            self.stop_record_button.setVisible(False)
            self.record_button.setVisible(True)
        else:
            print("Error: recognized_text is not a string")

    def display_sign_language_image(self, word):
        image_path = os.path.join("sign_language_images", word.lower() + ".jpg")
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = image_rgb.shape
                bytesPerLine = ch * w
                qImg = QImage(image_rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)
                qImg = qImg.scaled(200, 200, Qt.KeepAspectRatio)
                self.sign_language_label.setPixmap(QPixmap.fromImage(qImg))
        else:
            print("Sign language image not found for word:", word)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = YOLO_GUI()
    mainWindow.show()
    sys.exit(app.exec_())
