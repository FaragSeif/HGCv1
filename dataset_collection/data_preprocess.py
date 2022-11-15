from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

class DataPreprocess:
    def __init__(self):
        pass

    def data_preprocess():
        data_path = os.path.join("dataset")
        actions = np.array(["front", "back", "right", "left", "stop"])
        no_sequences = 5
        sequence_length = 5

        label_map = {label:num for num, label in enumerate(actions)}
        print(label_map)

        sequences , labels = [], []
        for action in actions: 
            for sequence in range(no_sequences):
                for frame_num  in range(sequence_length):
                    res = np.load(os.path.join("dataset", action, str(sequence)+str(frame_num)+".npy"))
                    sequences.append(res)  
                    labels.append(label_map[action])

        X = np.array(sequences, dtype=object)
        enc = OneHotEncoder(handle_unknown='ignore')
        y = enc.fit_transform(np.array(labels).reshape(-1,1)).toarray()
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.05)

        return X_train, X_test, Y_train, Y_test
