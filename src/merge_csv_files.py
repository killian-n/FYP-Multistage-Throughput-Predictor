import pandas as pd
import os

OUTER_FOLDER_PATH = "C:/Users/knola/Desktop/Final Year Project/Datasets/Raw/4G_LTE/"


def merge_csv_files(path_to_folder):
    folder_names = os.listdir(path_to_folder)
    raw_data = pd.DataFrame()
    session_value = 0
    for name in folder_names:
        movement_type = name
        file_directory = path_to_folder + name + "/"
        csv_names = os.listdir(file_directory)
        for file in csv_names:
            file_path = file_directory + file
            new_data = pd.read_csv(file_path, index_col=None)
            new_data["movement_type"] = movement_type
            new_data["session"] = session_value
            raw_data = pd.concat([raw_data, new_data], axis=0)
            session_value += 1
    return raw_data

if __name__ == "__main__":
    raw_df = merge_csv_files(OUTER_FOLDER_PATH)
    merged_file_path = "C:/Users/knola/Desktop/Final Year Project/Datasets/Raw/all_4G_data.csv"
    raw_df.to_csv(merged_file_path, index=False, encoding="UTF-8")




