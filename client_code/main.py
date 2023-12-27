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
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib import pyplot as plt
import analyse_result
import clip_eye_with_mediapipe
import configs
import exp
import clip_with_yolo
import image_to_tensor
import log_to_tensor
import regression_with_resnet
import util_functions


'''
训练结果备注：
ver 0, 1, 2: 固定learning rate为0.002。epoch总数为100。对应best-checkpoint.ckpt。这里似乎出现了梯度爆炸的情况。
ver 3, 4, 5: learning rate起始为0.001，每过10个epoch后乘以0.1。epoch总数为100。对应best-checkpoint-v1.ckpt。目前看这个模型还没有收敛。
ver 6, 7, 8: learning rate起始为0.002，每过10个epoch后乘以0.8。epoch总数为100。对应best-checkpoint-v2.ckpt。
'''


if __name__ == '__main__':
    # exp.run_exp()

    # 检查图像数量是否都至少大于20。
    # utils.check_file_num(20, "0.0")
    # utils.check_file_num(20, "0.1")

    # 从完整图片中切出人的脸部。
    # clip_with_yolo.clip_human_in_batch("0.0")
    # clip_with_yolo.clip_human_in_batch("0.1")

    # 从完整图片中中切出眼睛。
    # clip_eye_with_mediapipe.clip_eyes_of_subject("0.0")
    # clip_eye_with_mediapipe.clip_eyes_of_subject("0.1")

    # 将image转换为tensor。
    # image_to_tensor.image_to_tensor("clipped_camera_distant", 300)
    # image_to_tensor.image_to_tensor("clipped_camera_left_eye", 300)
    # image_to_tensor.image_to_tensor("clipped_camera_right_eye", 300)
    # image_to_tensor.image_to_tensor("camera_left_eye", 300)
    # image_to_tensor.image_to_tensor("camera_right_eye", 300)

    # 将log转换为tensor。
    # log_to_tensor.log_to_tensor("log_clip")
    # log_to_tensor.log_to_tensor("log_clip_eye")

    # 进行resnet模型的训练。
    # regression_with_resnet.train_model_in_batches(1, None, "all_data")
    # regression_with_resnet.train_model_in_batches(1, None, "eye_camera_only")
    # regression_with_resnet.train_model_in_batches(1, None, "distant_camera_and_distant_eye")

    # 验证模型结果。
    model_name = "best-checkpoint-v2.ckpt"
    all_data_loss_list, all_data_actual_loss_list, all_data_prediction_and_truth_list \
        = regression_with_resnet.validate_data_of_subject(0, model_name, "all_data")
    eye_camera_only_loss_list, eye_camera_only_actual_loss_list, eye_camera_only_prediction_and_truth_list \
        = regression_with_resnet.validate_data_of_subject(0, model_name, "eye_camera_only")
    distant_camera_and_distant_eye_loss_list, distant_camera_and_distant_eye_actual_loss_list, distant_camera_and_distant_eye_prediction_and_truth_list \
        = regression_with_resnet.validate_data_of_subject(0, model_name, "distant_camera_and_distant_eye")

    # 可视化最终结果。
    analyse_result.visualize_error_over_the_map(all_data_actual_loss_list, all_data_prediction_and_truth_list)
    # analyse_result.visualize_error_over_the_map(0, "best-checkpoint-v1.ckpt", "eye_camera_only")
    # analyse_result.visualize_error_over_the_map(0, "best-checkpoint-v1.ckpt", "distant_camera_and_distant_eye")

    # TODO 用部分数据训练模型，然后用全部数据进行测试。

