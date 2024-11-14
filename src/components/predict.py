import numpy as np
import os
import pickle

from src.components import data_ingestion
from src.components import 

class PredictModuleConfig:
    artifacts = os.path.join(os.getcwd(), "artifacts")
    transformation_object_file = "trasformation.pkl"
    predict_file = "data/raw_data/111579.npz"

class PredictModule:

    def __init__(self, datapoint_df):
        self.predict_config = PredictModuleConfig()
        self.df = datapoint_df

    def performTransformation(self):
        transformer_obj_file = os.path.join(self.predict_config.artifacts, self.predict_config.transformer_obj_file)
        transfrmation_obj = pickle.load(transformer_obj_file)
        data = np.load(os.path.join(os.getcwd(), self.predict_config.predict_file))
        preprocessed_data = data_ingestion.DataIngestion.preprocess_file(data)
        preprocessed_data = np.array(preprocessed_data)
        data_df = pd.DataFrame(preprocessed_data, columns=[
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
        transformed_data = transfrmation_obj.transform(data_df)
        
        

        
if __name__ == "__main__":
    