import os


def get_row_and_col(class_name):
    row_val, col_val = map(float, class_name.replace("row_", "").replace("col_", "").split('-'))
    return row_val, col_val


def check_file_num(num_limit, subject_num):
    file_path_prefix = f"output/{subject_num}/"
    camera_distant_file_prefix = f"{file_path_prefix}camera_distant/"
    camera_left_eye_file_prefix = f"{file_path_prefix}camera_left_eye/"
    camera_right_eye_file_prefix = f"{file_path_prefix}camera_right_eye/"

    check_file_num_given_path(num_limit, camera_distant_file_prefix)
    check_file_num_given_path(num_limit, camera_left_eye_file_prefix)
    check_file_num_given_path(num_limit, camera_right_eye_file_prefix)


def check_file_num_given_path(num_limit, file_prefix):
    file_list = os.listdir(file_prefix)
    file_list.sort(key=get_row_and_col)

    for file_index, file_name in enumerate(file_list):
        file_prefix = f"{file_prefix}/{file_name}"
        image_file_list = os.listdir(file_prefix)
        if len(image_file_list) < num_limit:
            print(f"{file_prefix} has {len(image_file_list)} files")

