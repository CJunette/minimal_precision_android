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
import utils

if __name__ == '__main__':
    # exp.run_exp()
    utils.check_file_num(20, "0.0")
    # utils.check_file_num(20, "0.1")

