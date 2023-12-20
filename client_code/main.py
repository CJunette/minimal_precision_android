import os
import socket
import random

import cv2
import keyboard
from functools import partial
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import configs


def init_name_list():
    name_list = []
    for i in range(configs.row_num):
        for j in range(configs.col_num):
            name_list.append({"circle_index": i * configs.col_num + j, "name": f"row_{i}-col_{j}"})
    random.shuffle(name_list)
    return name_list


class CommunicationThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, client_socket):
        super().__init__()
        self.client_socket = client_socket

    def run(self):
        while True:
            data = self.client_socket.recv(1024).decode('utf-8')
            if data:
                print(f"Received: {data}")
                self.update_signal.emit(data)


class CaptureVideoThread(QThread):
    def __init__(self):
        super().__init__()
        self.bool_capture = False
        self.capture_index = 0
        self.file_path = None
        self.cap = cv2.VideoCapture(0)
        # show cap result
        # while True:
        #     ret, frame = self.cap.read()
        #     if ret:
        #         cv2.imshow("capture", frame)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        # self.cap.release()

    def run(self):
        while True:
            while self.bool_capture and self.file_path:
                file_path = f"output/subject_{configs.subject_num}/{configs.mode}/{self.file_path}/"
                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                ret, frame = self.cap.read()
                if ret:
                    filename = f"{file_path}/capture_{self.capture_index}.jpg"
                    cv2.imwrite(filename, frame)
                    self.capture_index += 1

    def capture_video(self, data):
        data_split = data.split("*")
        if data_split[0] == "start":
            self.bool_capture = True
            self.file_path = data_split[1]
        else:
            self.bool_capture = False
            self.file_path = None
            self.capture_index = 0


class MainWindow(QMainWindow):
    update_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Key Press Example")
        self.setGeometry(100, 100, 600, 400)
        self.label = QLabel("Press any key", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)
        self.client_socket = None

        self.host = 'localhost'
        self.port = 12345
        self.client_socket = socket.socket()
        # self.client_socket.connect((self.host, self.port))
        self.client_socket.connect((self.host, self.port))
        message = "hello from python\n"
        self.client_socket.send(message.encode())

        self.name_list = init_name_list()
        self.current_index = -1
        self.bool_exp_processing = False

        self.communication_thread = CommunicationThread(self.client_socket)
        self.communication_thread.update_signal.connect(self.on_receive_data)
        self.communication_thread.start()

        self.capture_video_thread = CaptureVideoThread()
        self.update_signal.connect(self.capture_video_thread.capture_video)
        self.capture_video_thread.start()

    def on_receive_data(self, data):
        if data == "finish":
            self.bool_exp_processing = False
            self.update_signal.emit("finish*")

    def keyPressEvent(self, event):
        self.label.setText(event.text())

        if self.client_socket and not self.bool_exp_processing:
        # if True:
            message = None # message: event, current_index, current_circle_index, current_circle_name, next_circle_index
            bool_record_video = False
            if event.key() == Qt.Key_Escape:
                print("exit")
                self.client_socket.send("exit\n".encode())
                self.client_socket.close()
            else:
                if event.key() == Qt.Key_S or event.key() == Qt.Key_W:
                    if event.key() == Qt.Key_S:
                        if self.current_index == len(self.name_list):
                            return
                        else:
                            if self.current_index == -1:
                                message = (f"{event.text()}*"
                                           f"{self.current_index}*"
                                           f"null*"
                                           f"null*"
                                           f"{self.name_list[0].get('circle_index')}\n")
                            elif self.current_index < len(self.name_list) - 1:
                                message = (f"{event.text()}*"
                                           f"{self.current_index}*"
                                           f"{self.name_list[self.current_index].get('circle_index')}*"
                                           f"{self.name_list[self.current_index].get('name')}*"
                                           f"{self.name_list[self.current_index + 1].get('circle_index')}\n")
                            elif self.current_index == len(self.name_list) - 1:
                                message = (f"{event.text()}*"
                                           f"{self.current_index}*"
                                           f"{self.name_list[self.current_index].get('circle_index')}*"
                                           f"{self.name_list[self.current_index].get('name')}*"
                                           f"null\n")
                            else:
                                # 这个循环不会被打倒，因为上面的if self.current_index == len(self.name_list)已经处理了这个情况。
                                # 这里写它是为了让上面的条件看起来更加清晰。
                                return

                            if -1 < self.current_index <= len(self.name_list) - 1:
                                bool_record_video = True
                            self.current_index += 1

                    elif event.key() == Qt.Key_W:
                        if self.current_index == -1:
                            return
                        else:
                            self.current_index -= 1
                            if self.current_index == -1:
                                self.current_index = 0
                                return
                            elif self.current_index == len(self.name_list) - 1:
                                message = (f"{event.text()}*"
                                           f"{self.current_index}*"
                                           f"{self.name_list[self.current_index].get('circle_index')}*"
                                           f"{self.name_list[self.current_index].get('name')}*"
                                           f"null\n")
                            else:
                                message = (f"{event.text()}*"
                                           f"{self.current_index}*"
                                           f"{self.name_list[self.current_index].get('circle_index')}*"
                                           f"{self.name_list[self.current_index].get('name')}*"
                                           f"{self.name_list[self.current_index + 1].get('circle_index')}\n")

                    print("sent: ", message)
                    self.client_socket.send(message.encode())

                    self.bool_exp_processing = True
                    if bool_record_video:
                        self.update_signal.emit(f"start*{self.name_list[self.current_index].get('name')}")


    # def capture_video(self):
    #     file_path = f"output/subject_{configs.subject_num}/{configs.mode}/{self.name_list[self.current_index]}/"
    #     if not os.path.exists(file_path):
    #         os.makedirs(file_path)
    #
    #     for i in range(20):
    #         ret, frame = self.cap.read()
    #         if ret:
    #             filename = f"{file_path}/capture_{i}.jpg"
    #             cv2.imwrite(filename, frame)


if __name__ == '__main__':
    # server()
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())


