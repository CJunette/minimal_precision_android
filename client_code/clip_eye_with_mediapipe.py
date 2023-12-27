import os

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import configs
import util_functions


def clip_eyes(holistic, image):
    results = holistic.process(image)
    left_eye_indices = [417, 282, 353, 348]
    right_eye_indices = [124, 52, 193, 119]

    left_eye_coordinates = []
    right_eye_coordinates = []

    if results.face_landmarks is None:
        # show image use plt
        plt.imshow(image)
        plt.show()
        return None, None

    for idx in left_eye_indices:
        landmark = results.face_landmarks.landmark[idx]
        height, width, _ = image.shape
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        left_eye_coordinates.append((cx, cy))
    for idx in right_eye_indices:
        landmark = results.face_landmarks.landmark[idx]
        height, width, _ = image.shape
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        right_eye_coordinates.append((cx, cy))

    return left_eye_coordinates, right_eye_coordinates


def save_clipped_eye(file_path, image_name, holistic, log_file, subject_name, col_row_file_name):
    image_path = f"{file_path}/{image_name}"
    image = cv2.imread(image_path)

    left_eye_coordinates, right_eye_coordinates = clip_eyes(holistic, image)
    if left_eye_coordinates is None and right_eye_coordinates is None:
        log_file.write(f"no eyes detected in {image_path}\n")
        print(f"no eyes detected in {image_path}\n")
        return

    left_mask = [[left_eye_coordinates[0][0], left_eye_coordinates[1][1]],
                 [left_eye_coordinates[2][0], left_eye_coordinates[1][1]],
                 [left_eye_coordinates[2][0], left_eye_coordinates[3][1]],
                 [left_eye_coordinates[0][0], left_eye_coordinates[3][1]]]
    right_mask = [[right_eye_coordinates[0][0], right_eye_coordinates[1][1]],
                  [right_eye_coordinates[2][0], right_eye_coordinates[1][1]],
                  [right_eye_coordinates[2][0], right_eye_coordinates[3][1]],
                  [right_eye_coordinates[0][0], right_eye_coordinates[3][1]]]

    result_left = image[left_mask[1][1]:left_mask[3][1], left_mask[3][0]:left_mask[1][0]]
    result_right = image[right_mask[1][1]:right_mask[3][1], right_mask[3][0]:right_mask[1][0]]
    height = 150
    result_left = cv2.resize(result_left, (int(result_left.shape[1] * height / result_left.shape[0]), height))
    result_right = cv2.resize(result_right, (int(result_right.shape[1] * height / result_right.shape[0]), height))

    output_size = (300, 300)
    output_image = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
    output_image[0:result_left.shape[0], 0:result_left.shape[1]] = result_left
    output_image[output_image.shape[0] - result_right.shape[0]:output_image.shape[0], 0:result_right.shape[1]] = result_right

    # cv2.imshow("MediaPipe Holistic", output_image)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break

    output_path = f"output/subject_{subject_name}/{configs.mode}_eye/{col_row_file_name}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(f"{output_path}{image_name}", output_image)


def interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value, next_value):
    result = (current_index - previous_index) * (next_value - previous_value) / (next_index - previous_index) + previous_value
    return result


def get_interpolate_result(result_list, current_index, previous_index, next_index):
    previous_value = result_list[previous_index]
    next_value = result_list[next_index]

    left_current_0_0 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[0][0][0], next_value[0][0][0])
    left_current_0_1 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[0][0][1], next_value[0][0][1])
    left_current_1_0 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[0][1][0], next_value[0][1][0])
    left_current_1_1 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[0][1][1], next_value[0][1][1])
    left_current_2_0 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[0][2][0], next_value[0][2][0])
    left_current_2_1 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[0][2][1], next_value[0][2][1])
    left_current_3_0 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[0][3][0], next_value[0][3][0])
    left_current_3_1 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[0][3][1], next_value[0][3][1])
    left_current = [(left_current_0_0, left_current_0_1),
                    (left_current_1_0, left_current_1_1),
                    (left_current_2_0, left_current_2_1),
                    (left_current_3_0, left_current_3_1)]

    right_current_0_0 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[1][0][0], next_value[1][0][0])
    right_current_0_1 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[1][0][1], next_value[1][0][1])
    right_current_1_0 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[1][1][0], next_value[1][1][0])
    right_current_1_1 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[1][1][1], next_value[1][1][1])
    right_current_2_0 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[1][2][0], next_value[1][2][0])
    right_current_2_1 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[1][2][1], next_value[1][2][1])
    right_current_3_0 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[1][3][0], next_value[1][3][0])
    right_current_3_1 = interpolate_eye_coordinates(current_index, previous_index, next_index, previous_value[1][3][1], next_value[1][3][1])
    right_current = [(right_current_0_0, right_current_0_1),
                     (right_current_1_0, right_current_1_1),
                     (right_current_2_0, right_current_2_1),
                     (right_current_3_0, right_current_3_1)]

    current_value = [left_current, right_current]

    return current_value


def save_clipped_eye_of_same_size(file_path, image_names, holistic, log_file, subject_name, col_row_file_name, resize_height=500):
    '''
    修改了原有的函数，目前将一个row-col文件夹下的所有图片统一处理。
    :param file_path:
    :param image_names:
    :param holistic:
    :param log_file:
    :param subject_name:
    :param col_row_file_name:
    :param resize_height:
    :return:
    '''
    eye_coordinates_list = [[None, None] for _ in range(len(image_names))]

    bool_interpolate = False
    for image_index, image_name in enumerate(image_names):
        image_path = f"{file_path}/{image_name}"

        # check if image_path is in log_file["image_path"](pd.DataFrame) exist, if not create it
        if image_path not in log_file["image_path"].values:
            log_file.loc[len(log_file)] = [image_path, None, None, None, None, None, None, None, None, None]

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        left_eye_coordinates, right_eye_coordinates = clip_eyes(holistic, image_rgb)
        if left_eye_coordinates is None and right_eye_coordinates is None:
            # log_file.loc[file_path, "other"] = f"no eyes detected in {image_path}\n"
            print(f"no eyes detected in {image_path}\n")
            bool_interpolate = True
            log_file.loc["image_path" == image_path]["other"] = f"no eyes detected\n"

        eye_coordinates_list[image_index] = [left_eye_coordinates, right_eye_coordinates]
    if bool_interpolate:
        eye_coordinates_list = util_functions.interpolate_none(eye_coordinates_list, get_interpolate_result, [None, None])

    for eye_coordinates_index, eye_coordinates in enumerate(eye_coordinates_list):
        left_eye_coordinates = eye_coordinates[0]
        right_eye_coordinates = eye_coordinates[1]

        left_mask = [[left_eye_coordinates[0][0], left_eye_coordinates[1][1]],
                     [left_eye_coordinates[2][0], left_eye_coordinates[1][1]],
                     [left_eye_coordinates[2][0], left_eye_coordinates[3][1]],
                     [left_eye_coordinates[0][0], left_eye_coordinates[3][1]]]
        right_mask = [[right_eye_coordinates[0][0], right_eye_coordinates[1][1]],
                      [right_eye_coordinates[2][0], right_eye_coordinates[1][1]],
                      [right_eye_coordinates[2][0], right_eye_coordinates[3][1]],
                      [right_eye_coordinates[0][0], right_eye_coordinates[3][1]]]

        # 调整大小，使左右眼的裁切区域都是正方形。
        left_width = left_mask[1][0] - left_mask[3][0]
        left_y_center = (left_mask[3][1] + left_mask[1][1]) / 2
        new_left_y_min = left_y_center - left_width // 2
        new_left_y_max = new_left_y_min + left_width
        left_mask[0][1] = left_mask[1][1] = int(new_left_y_min)
        left_mask[2][1] = left_mask[3][1] = int(new_left_y_max)

        right_width = right_mask[1][0] - right_mask[3][0]
        right_y_center = (right_mask[3][1] + right_mask[1][1]) / 2
        new_right_y_min = right_y_center - right_width // 2
        new_right_y_max = new_right_y_min + right_width
        right_mask[0][1] = right_mask[1][1] = int(new_right_y_min)
        right_mask[2][1] = right_mask[3][1] = int(new_right_y_max)

        image_name = image_names[eye_coordinates_index]
        image_path = f"{file_path}/{image_name}"
        image = cv2.imread(image_path)

        # set log_file["image_path" == image_path]

        log_file.loc[log_file["image_path"] == image_path, "image_path"] = image_path
        log_file.loc[log_file["image_path"] == image_path, "left_start_x"] = left_mask[3][0]
        log_file.loc[log_file["image_path"] == image_path, "left_start_y"] = left_mask[1][1]
        log_file.loc[log_file["image_path"] == image_path, "left_end_x"] = left_mask[1][0]
        log_file.loc[log_file["image_path"] == image_path, "left_end_y"] = left_mask[3][1]
        log_file.loc[log_file["image_path"] == image_path, "right_start_x"] = right_mask[3][0]
        log_file.loc[log_file["image_path"] == image_path, "right_start_y"] = right_mask[1][1]
        log_file.loc[log_file["image_path"] == image_path, "right_end_x"] = right_mask[1][0]
        log_file.loc[log_file["image_path"] == image_path, "right_end_y"] = right_mask[3][1]

        result_left = image[left_mask[1][1]:left_mask[3][1], left_mask[3][0]:left_mask[1][0]]
        result_right = image[right_mask[1][1]:right_mask[3][1], right_mask[3][0]:right_mask[1][0]]
        result_left = cv2.resize(result_left, (resize_height, resize_height))
        result_right = cv2.resize(result_right, (resize_height, resize_height))

        # # plt 2*1 fig, show result_left and result_right
        # fig, axs = plt.subplots(2, 1)
        # fig.set_size_inches(64, 36)
        # fig.subplots_adjust(wspace=0.05, hspace=0.025, left=0.05, right=0.95, top=0.95, bottom=0.05)
        # axs[0].imshow(result_left)
        # axs[1].imshow(result_right)
        # plt.show()

        left_output_path = f"output/subject_{subject_name}/clipped_camera_left_eye/{col_row_file_name}"
        right_output_path = f"output/subject_{subject_name}/clipped_camera_right_eye/{col_row_file_name}"
        if not os.path.exists(left_output_path):
            os.makedirs(left_output_path)
        if not os.path.exists(right_output_path):
            os.makedirs(right_output_path)
        cv2.imwrite(f"{left_output_path}/{image_name}", result_left)
        cv2.imwrite(f"{right_output_path}/{image_name}", result_right)

    log_file.to_csv(f"log/log_clip_eye_{subject_name}.csv", index=False)


def clip_eyes_of_subject(subject_name: str):
    root_dir = f'output/subject_{subject_name}/camera_distant'
    file_path_list = os.listdir(root_dir)
    file_path_list.sort(key=util_functions.get_row_and_col)

    log_file_path = f"log/subject_{subject_name}"
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
        log_file_name = f"{log_file_path}/log_clip_eye.csv"
        log_file_header = ["image_path",
                           "left_start_x", "left_start_y", "left_end_x", "left_end_y",
                           "right_start_x", "right_start_y", "right_end_x", "right_end_y",
                           "other"]
        log_file = pd.DataFrame(columns=log_file_header)
        log_file.to_csv(log_file_name, index=False)

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for col_row_file_name in file_path_list:
            print(col_row_file_name)
            file_path = f"{root_dir}/{col_row_file_name}"
            image_names = os.listdir(file_path)
            image_names.sort(key=lambda x: int(x.replace(".jpg", "").replace("capture_", "")))

            log_file = pd.read_csv(log_file_path)
            save_clipped_eye_of_same_size(file_path, image_names, holistic, log_file, subject_name, col_row_file_name)  # 这里输出的是左右眼分开的图片。


def test():
    '''
    仅用于测试代码可用性。
    :return:
    '''
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(1)  # 0表示默认摄像头

    print(cap)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            # 将帧转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 运行Holistic模型来检测人眼
            results = holistic.process(frame_rgb)

            left_eye_indices = [417, 282, 353, 348]
            right_eye_indices = [124, 52, 193, 119]

            left_eye_coordinates = []
            right_eye_coordinates = []

            for idx in left_eye_indices:
                landmark = results.face_landmarks.landmark[idx]
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                left_eye_coordinates.append((cx, cy))

                # 绘制右眼的关键点
            for idx in right_eye_indices:
                landmark = results.face_landmarks.landmark[idx]
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                right_eye_coordinates.append((cx, cy))

            left_mask = [[left_eye_coordinates[0][0], left_eye_coordinates[1][1]],
                         [left_eye_coordinates[2][0], left_eye_coordinates[1][1]],
                         [left_eye_coordinates[2][0], left_eye_coordinates[3][1]],
                         [left_eye_coordinates[0][0], left_eye_coordinates[3][1]]]
            right_mask = [[right_eye_coordinates[0][0], right_eye_coordinates[1][1]],
                          [right_eye_coordinates[2][0], right_eye_coordinates[1][1]],
                          [right_eye_coordinates[2][0], right_eye_coordinates[3][1]],
                          [right_eye_coordinates[0][0], right_eye_coordinates[3][1]]]

            # mask = np.zeros_like(frame)
            # mask = cv2.fillPoly(mask, [np.array(left_mask)], (255, 255, 255))
            # mask = cv2.fillPoly(mask, [np.array(right_mask)], (255, 255, 255))
            # result = cv2.bitwise_and(frame, mask)

            # 这里在frame中把xy倒过来的原因是因为frame的shape是(height, width, channel)，而opencv的坐标系是(x, y)，所以要把x和y倒过来。
            result_left = frame[left_mask[1][1]:left_mask[3][1], left_mask[3][0]:left_mask[1][0]]
            result_right = frame[right_mask[1][1]:right_mask[3][1], right_mask[3][0]:right_mask[1][0]]
            output_size = (100, 100)
            output_image = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
            output_image[0:result_left.shape[0], 0:result_left.shape[1]] = result_left
            output_image[output_image.shape[0] - result_right.shape[0]:output_image.shape[0], 0:result_right.shape[1]] = result_right

            cv2.imshow("MediaPipe Holistic", output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
