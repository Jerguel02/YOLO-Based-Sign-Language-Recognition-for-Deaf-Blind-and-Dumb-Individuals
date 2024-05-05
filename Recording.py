import sys
import os
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt, QEventLoop, QMetaObject, Qt, Q_ARG, pyqtSignal
import threading
import wave
import pyaudio

class YOLO_GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Voice Recorder & Sign Language GUI")
        self.setGeometry(100, 100, 800, 600)

        self.sign_language_label = QLabel(self)
        self.sign_language_label.setAlignment(Qt.AlignCenter)

        self.recording_label = QLabel("Recording...", self)
        self.recording_label.setAlignment(Qt.AlignCenter)
        self.recording_label.setVisible(False)

        self.record_button = QPushButton("Record", self)
        self.record_button.setMaximumHeight(30) 
        self.record_button.clicked.connect(self.record_audio)

        self.stop_record_button = QPushButton("Stop", self)
        self.stop_record_button.setMaximumHeight(30)
        self.stop_record_button.setVisible(False)
        self.stop_record_button.clicked.connect(self.stop_record)

        self.text_display = QTextEdit(self)
        self.text_display.setReadOnly(True)
        self.text_display.setMaximumHeight(50)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.sign_language_label)
        main_layout.addWidget(self.recording_label)
        main_layout.addWidget(self.record_button)
        main_layout.addWidget(self.stop_record_button)
        main_layout.addWidget(self.text_display)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.sign_language_images_folder = "sign_language_images"
        self.audio_filename = "audio/AUDIO.wav"
        self.recognizer = sr.Recognizer()
        self.recording = False
        self.recognized_text = ""
        
    def record_audio(self):
        print("Recording...")
        self.text_display.clear()
        self.recording_label.setVisible(True)
        self.record_button.setVisible(False)
        self.stop_record_button.setVisible(True)
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_() 
        thread1.start()
        self._update_gui()


    def _record_audio_thread(self):
        CHUNK = 2048
        SAMPLE_FORMAT = pyaudio.paInt16
        CHANNELS = 1
        FS = 44100
        DURATION = 9
        FILE_NAME = self.audio_filename

        p = pyaudio.PyAudio()
        stream = p.open(format=SAMPLE_FORMAT,
                        channels=CHANNELS,
                        rate=FS,
                        frames_per_buffer=CHUNK,
                        input=True)
        frames = []
        self.recording = True
        for _ in range(0, int(FS / CHUNK * DURATION)):
            if self.recording:
                data = stream.read(CHUNK)
                frames.append(data)
            if not self.recording:
                break
        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(FILE_NAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
            wf.setframerate(FS)
            wf.writeframes(b''.join(frames))

        print("Recording saved as", self.audio_filename)
        self.recording = False

    def _update_gui(self):
        self.recognized_text = self.recognize_speech(self.audio_filename)
        self.text_display.setPlainText(self.recognized_text)
        self.display_sign_language_image(self.recognized_text)
        self.recognized_text = ""
        self.recording_label.setVisible(False)
        self.stop_record_button.setVisible(False)
        self.record_button.setVisible(True)

    def stop_record(self):
        self.recording = False
        self.recording_label.setVisible(False)
        self.stop_record_button.setVisible(False)
        self.record_button.setVisible(True)



    def recognize_speech(self, filename):
        with sr.AudioFile(filename) as source:
            audio_data = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio_data)
                print("Recognized text:", text)
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                return ""
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                return ""

    def display_sign_language_image(self, word):
        image_path = os.path.join(self.sign_language_images_folder, word.lower() + ".jpg")

        if os.path.exists(image_path):
            image = cv2.imread(image_path)
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
