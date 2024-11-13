import os
import sys
import numpy as np
import pandas as pd
import torch
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from zipfile import BadZipFile

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from src import utils
from src import constants

@dataclass
class DataIngestionConfig:
    raw_data_path: str = "data/raw_data"


class DataIngestion:
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_dict = {}

    def initiate_data_ingestion(self):
            dataset = []
            logging.info("Data Ingestion Started.")
            count = 0
            for filename in tqdm(os.listdir(os.path.join(os.getcwd(), self.data_ingestion_config.raw_data_path))):
                data_point = []
                if filename.endswith('.npz') and os.path.getsize(os.path.join(os.getcwd(), self.data_ingestion_config.raw_data_path, filename)):
                    try:
                        file_path = os.path.join(os.getcwd(), self.data_ingestion_config.raw_data_path, filename)
                        data = np.load(file_path, allow_pickle=True)
                        if "metadata" not in data.keys():
                            print(f"{filename} skipped.")
                            continue
                        genre = utils.preprocess_genres(data["metadata"][0]["genres"])
                        if not genre:
                            print(f"{filename} skipped.")
                            continue
                        data_point.append(genre)
                        data_point.append(data["metadata"][0]["bit_rate"])
                        data_point.append(data["metadata"][0]["duration"])
                        data_point.append(utils.categorize_listens(data["metadata"][0]["listens"]))
                        
    
                        data = utils.reshape_all_time_series_data(data)
                        data_point.append(np.array(data["mel_spectrogram"]))
                        data_point.append(np.array(data["mfccs"]))
                        data_point.append(np.array(data["chroma"]))
                        data_point.append(np.array(data["spectral_contrast"]))
                        data_point.append(np.array(data["zcr"]))
                        data_point.append(np.array(data["spectral_centroid"]))
                        data_point.append(np.array(data["spectral_bandwidth"]))
                        data_point.append(np.array(data["rms_energy"]))
                        data_point.append(np.array(data["tonnetz"]))
                        print(len(data_point))
                        dataset.append(data_point)
                        count += 1
                        print(f"Total Files Processed: {count}")
                    except (EOFError, OSError, ValueError, BadZipFile) as e:
                        print(f"{filename} skipped.")
                        # raise CustomException(e, sys)
              
            dataset = np.array(dataset, dtype=object)
            dataset_df = pd.DataFrame(dataset, columns=[
                "genre",
                "bit_rate",
                "duration",
                "success",
                "mel_spectrogram",
                "mfccs",
                "chroma",
                "spectral_contrast",
                "zcr",
                "spectral_centroid",
                "spectral_bandwidth",
                "rms_energy",
                "tonnetz",
                ])
            logging.info("Data Ingestion Finished.")
            return dataset_df
