import sys
import os
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt, QEventLoop
import threading
#import ffmpeg
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

        #self.audio_filename_input = "audio/AUDIO.mp3"
        self.sign_language_images_folder = "sign_language_images"
        self.audio_filename = "audio/AUDIO.wav"
        self.recognizer = sr.Recognizer()
    def record_audio(self):
        print("Recording...")
        self.recording_label.setVisible(True)
        self.record_button.setVisible(False)
        self.stop_record_button.setVisible(True)
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_()
        #self.recording_label.setVisible(True)
        CHUNK = 1024
        SAMPLE_FORMAT = pyaudio.paInt16
        CHANNEL = 1     
        FS = 44100
        DURATION = 5
        FILE_NAME = self.audio_filename
        #thread = threading.Thread(target=self._record_audio_thread, args=(fs, duration))
        thread = threading.Thread(target = self._record_audio_thread, args = (CHUNK, SAMPLE_FORMAT, CHANNEL, FS, DURATION, FILE_NAME))
        thread.start()

    """"def check_and_convert_to_pcm_wav(self, input_filename):
        output_filename = "audio/AUDIO.wav"
        try:
            # Mở tệp âm thanh đầu vào để kiểm tra thông tin định dạng
            with wave.open(input_filename, 'rb') as wf:
                # Kiểm tra định dạng của tệp âm thanh
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != 'NONE':
                    print("Input audio file is not in PCM WAV format.")
                    return False
        except Exception as e:
            print("Error occurred while opening input audio file:", e)
            return False
    
        try:
            # Mở tệp âm thanh đầu vào và đầu ra để chuyển đổi
            with wave.open(input_filename, 'rb') as infile, wave.open(output_filename, 'wb') as outfile:
                # Lấy thông tin âm thanh từ tệp đầu vào và đặt nó cho tệp đầu ra
                outfile.setparams(infile.getparams())
                # Đọc dữ liệu âm thanh từ tệp đầu vào và ghi vào tệp đầu ra
                data = infile.readframes(infile.getnframes())
                outfile.writeframes(data)
            print("Audio file successfully converted to PCM WAV format.")
            self.audio_filename = "audio/AUDIO.wav"
            return True
        except Exception as e:
            print("Error occurred while converting audio file:", e)
            return False"""
    #def _record_audio_thread(self, fs, duration):
    def _record_audio_thread(self, CHUNK, SAMPLE_FORMAT, CHANNEL, FS, DURATION, FILE_NAME):
        #my_recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        #my_recording = sd.rec(duration * fs, samplerate=fs, channels=1, dtype = 'float64')
        #sd.wait()
        #write(self.audio_filename, fs, my_recording)
        p = pyaudio.PyAudio()
        stream = p.open(format = SAMPLE_FORMAT,
                        channels = CHANNEL,
                        rate = FS,
                        frames_per_buffer = CHUNK,
                        input = True)
        frames = []
        for i in range(0, int(FS/CHUNK*DURATION)):
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate
        print("Recording saved as", self.audio_filename)

        wf = wave.open(FILE_NAME, 'wb')
        wf.setchannels(CHANNEL)
        wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
        wf.getframerate(fs)
        wf.writeframe(b''.join(frames))
        wf.close()
        #self.check_and_convert_to_pcm_wav(self.audio_filename)
        #ffmpeg.input(self.audio_filename_input).output(self.audio_filename).run()
        recognized_text = self.recognize_speech(self.audio_filename)

        self.text_display.setPlainText(recognized_text)

        self.display_sign_language_image(recognized_text)
        self.recording_label.setVisible(False)
        self.stop_record_button.setVisible(False)
        self.record_button.setVisible(True)

    
    
    def stop_record(self):
        sd.stop()
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_()
        print("Recording stopped")
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
