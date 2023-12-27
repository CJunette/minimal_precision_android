import os

import configs


def get_row_and_col(class_name):
    row_val, col_val = map(float, class_name.replace("row_", "").replace("col_", "").split('-'))
    return row_val, col_val


def check_file_num(num_limit, subject_num):
    file_path_prefix = f"output/subject_{subject_num}/"
    camera_distant_file_prefix = f"{file_path_prefix}camera_distant"
    camera_left_eye_file_prefix = f"{file_path_prefix}camera_left_eye"
    camera_right_eye_file_prefix = f"{file_path_prefix}camera_right_eye"

    check_file_num_given_path(num_limit, camera_distant_file_prefix)
    check_file_num_given_path(num_limit, camera_left_eye_file_prefix)
    check_file_num_given_path(num_limit, camera_right_eye_file_prefix)


def check_file_num_given_path(num_limit, file_prefix):
    file_list = os.listdir(file_prefix)
    file_list.sort(key=get_row_and_col)

    for file_index, file_name in enumerate(file_list):
        file_path = f"{file_prefix}/{file_name}"
        image_file_list = os.listdir(file_prefix)
        if len(image_file_list) < num_limit:
            print(f"{file_path} has {len(image_file_list)} files")


def interpolate_none(result_list, get_interpolate_result, none_format=None):
    '''
    该算法有个假设，就是result_list不会出现全都是None的情况。
    :param result_list:
    :param none_format:
    :return:
    '''
    check_none_index = 0
    while check_none_index < len(result_list):
        if result_list[check_none_index] is none_format:
            # 开头就是None的情况，找到后面的首个不是None的情况。
            if check_none_index == 0:
                next_index = 1
                while next_index < len(result_list):
                    if result_list[next_index] is not none_format:
                        for i in range(check_none_index, next_index):
                            result_list[i] = result_list[next_index]
                        break
                    next_index += 1
                check_none_index = next_index
            # 如果最后一个元素是None，由于筛选、修改是从前向后的，因此只要将前一个元素的值赋给最后一个元素即可。
            elif check_none_index == len(result_list) - 1:
                result_list[check_none_index] = result_list[check_none_index - 1]
            # 其余的情况即中间出现一个None。
            else:
                next_index = check_none_index + 1
                previous_index = check_none_index - 1
                while next_index <= len(result_list):
                    # 如果到末尾之前，出现了一个不是None的元素，则做线性插值。
                    if next_index < len(result_list) and result_list[next_index] is not None:
                        for i in range(previous_index + 1, next_index):
                            result_list[i] = get_interpolate_result(result_list, i, previous_index, next_index)
                        break
                    # 如果到末尾了，那么就将前面的所有None都赋值为前一个元素的值。
                    elif next_index == len(result_list):
                        for i in range(check_none_index, next_index):
                            result_list[i] = result_list[previous_index]
                        break
                    else:
                        next_index += 1
                check_none_index = next_index
        else:
            check_none_index += 1
    return result_list


def set_configs_bool_camera(bool_clipped_distant_camera,
                            bool_clipped_distant_left_eye_camera,
                            bool_clipped_distant_right_eye_camera,
                            bool_left_eye_camera,
                            bool_right_eye_camera):
    configs.bool_clipped_distant_camera = bool_clipped_distant_camera
    configs.bool_clipped_distant_left_eye_camera = bool_clipped_distant_left_eye_camera
    configs.bool_clipped_distant_right_eye_camera = bool_clipped_distant_right_eye_camera
    configs.bool_left_eye_camera = bool_left_eye_camera
    configs.bool_right_eye_camera = bool_right_eye_camera


def set_configs_bool_camera_given_mode(mode: str):
    if mode == "all_data":
        set_configs_bool_camera(1, 1, 1, 1, 1)
    elif mode == "eye_camera_only":
        set_configs_bool_camera(0, 0, 0, 1, 1)
    elif mode == "distant_camera_and_distant_eye":
        set_configs_bool_camera(1, 1, 1, 0, 0)

