import os
import torch
import numpy as np
import sys 
import pickle
# sys.modules['__main__.ArrayColumnTransformer'] = ArrayColumnTransformer
from src.components import model_trainer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src import utils
from src.components.array_column_transformer import ArrayColumnTransformer

class PredictPipeline:
    def __init__(self) -> None:
        self.model_path = os.path.join(os.getcwd(), "artifacts", "MODELS/epoch=59-step=6180.ckpt")
        self.transformer_obj_path = os.path.join(os.getcwd(), "artifacts", "transformation.pkl")
    
    def predict(self, data):
        # criterion = nn.CrossEntropyLoss()
        # learning_rate = 0.001
        # dropout_prob = 0.3
        checkpoint = torch.load(self.model_path,
                                map_location=lambda storage, loc: storage, weights_only=True)
        model = model_trainer.MusicSuccessPredictor()
        model.load_state_dict(state_dict=checkpoint["state_dict"])
        transform_obj = utils.load_object(self.transformer_obj_path)
        data_df = utils.process_music(data)

        transformed_data = transform_obj.transform(data_df)
        
        transformed_data = np.column_stack((data_df["genre"].values, transformed_data))

        transformed_data = utils.convert_dataset_into_tensor_dict(transformed_data)
        mel_spectrogram = transformed_data['mel_spectrogram'].unsqueeze(0)
        mfccs = transformed_data['mfccs'].unsqueeze(0)
        chroma = transformed_data['chroma'].unsqueeze(0)
        spectral_contrast = transformed_data['spectral_contrast'].unsqueeze(0)
        tonnetz = transformed_data['tonnetz'].unsqueeze(0)
        zcr = transformed_data['zcr']          # Shape: (batch_size, 1, 937)
        spectral_centroid = transformed_data['spectral_centroid'] # Shape: (batch_size, 1, 937)
        spectral_bandwidth = transformed_data['spectral_bandwidth'] # Shape: (batch_size, 1, 937)
        rms_energy = transformed_data['rms_energy']         # Shape: (batch_size, 1, 937)
        
        scalar_features = torch.stack((
                transformed_data['bit_rate'].float(),   # Scalar (batch_size, 1)
                transformed_data['duration'].float(),   # Scalar (batch_size, 1)
                transformed_data['genre'].float()      # Categorical scalar (batch_size, 1)
            ), dim=1)
        output = model(mel_spectrogram, mfccs, chroma, spectral_contrast, tonnetz, zcr, spectral_centroid, spectral_bandwidth, rms_energy, scalar_features)

        one_hot_output = torch.zeros(size=(1, 3))

        one_hot_output[:,torch.argmax(output)] = 1

        output_transformer = transform_obj.named_transformers_['cat_transforms']

        result = output_transformer.inverse_transform(one_hot_output)

        return result


        


