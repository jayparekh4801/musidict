import numpy as np
import pandas as pd
import os
import pickle
import sys
import torch

from src.exception import CustomException
from src.logger import logging
from src import constants


def categorize_listens(val):
    if constants.LISTENS_SPLIT[0] <= val < constants.LISTENS_SPLIT[1]:
        return "flop"
    elif constants.LISTENS_SPLIT[1] <= val < constants.LISTENS_SPLIT[2]:
        return "average"
    elif constants.LISTENS_SPLIT[2] <= val < constants.LISTENS_SPLIT[3]:  # Last boundary is np.inf, no upper limit
        return "hit"


def preprocess_genres(val):
    for i in val:
        if i in constants.NUMBER_TO_GENRE_MAPPING.keys():
            return i
    
    return None

def normalize_array_shape(array, target_shape):
    # Truncate if the array is larger than the target shape
    truncated_array = array[:target_shape[0], :target_shape[1]]
    
    # Pad if the array is smaller than the target shape
    padding = ((0, max(0, target_shape[0] - truncated_array.shape[0])),
               (0, max(0, target_shape[1] - truncated_array.shape[1])))
    
    normalized_array = np.pad(truncated_array, padding, mode='constant')
    return normalized_array


def reshape_all_time_series_data(data):
    reshaped_data = {}
    for key, val in data.items():
        if key == "original_audio":
            continue
        if key == "metadata":
            break
        if key in constants.TIME_SERIES_DATA_SHAPES.keys():
            reshaped_data[key] = normalize_array_shape(val, constants.TIME_SERIES_DATA_SHAPES[key])
    
    return reshaped_data


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        print(dir_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def convert_dataset_into_tensor_dict(data):
    success = torch.tensor(data[:, -3:].astype(float))
    result_dict = {}
    for ind, col in enumerate(constants.FEATURE_COLUMNS):
        # if col in  constants.TIME_SERIES_DATA_SHAPES.keys() and constants.TIME_SERIES_DATA_SHAPES[col][0] > 1:
        #     result_dict[col] = torch.tensor(np.stack(data[:, ind]), dtype=torch.float).unsqueeze(1)
        # else:
        print(data[:, ind])
        result_dict[col] = torch.tensor(np.stack(data[:, ind]), dtype=torch.float)

    
    result_dict["success"] = success

    return result_dict
