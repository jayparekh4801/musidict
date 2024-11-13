import numpy as np
import os

if __name__ == "__main__":
    file_dest = os.path.join(os.getcwd(), "data/transformed_data/transformed_dataset.npy")
    # print(os.getcwd())
    data = np.load(file_dest, allow_pickle=True)