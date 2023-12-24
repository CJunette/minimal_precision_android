import os
import shutil
import socket
import random
import subprocess
import time
import cv2
import keyboard
from functools import partial
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import configs
import exp
import clip_with_yolo


if __name__ == '__main__':
    # exp.run_exp()

    # 检查图像数量是否都至少大于20。
    # utils.check_file_num(20, "0.0")
    # utils.check_file_num(20, "0.1")

    clip_with_yolo.clip_human_in_batch("0.0")
    clip_with_yolo.clip_human_in_batch("0.1")

