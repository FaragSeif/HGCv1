import os
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


class KPClassifier:
    """
    A class to train, predict, save, and load gesture classifier model for mediapipe keypoints
    parameters:
        NUM_LABELS: number of classes to classify
        NUM_LABELS: number of classes to classify
    """

    def __init__(
        self,
        model_path=None,
        NUM_LABELS=7,
    ):
        self.trained = False
        self.model = None
        if model_path is None:
            self.model = models.Sequential(
                [
                    layers.Input((21 * 2,)),
                    layers.Dropout(0.0),
                    layers.Dense(32, activation="relu"),
                    layers.Dropout(0.0),
                    layers.Dense(32, activation="relu"),
                    layers.Dropout(0.0),
                    layers.Dense(16, activation="relu"),
                    layers.Dense(NUM_LABELS, activation="softmax"),
                ]
            )
            logging.info(" New model created")
            self.summary()
        else:
            # extention = model_path.split(".")[-1]
            # if extention == "hdf5":
            #     self.load(model_path)
            logging.info(' Loading TFLite model from "{}"'.format(model_path))
            self.load(model_path)

    def load(self, model_path, num_threads=1):
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def train(
        self,
        model_save_path,
        tflite_save_path,
        train_data,
        train_labels,
        val_data,
        val_labels,
        epochs=1000,
        batch_size=64,
        patience=10,
    ):
        """Start training the mode on given data

        Args:
            model_save_path (str): The path to save the model
            tflite_save_path (str): The path to save quantized tflite model used for inference
            train_data ([np.float32]): Input data for training
            train_labels ([np.float32]): Labels for training
            val_data ([np.float32]): Validation data
            val_labels ([np.float32]): Validation labels
            epochs (int, optional): Number of Epochs. Defaults to 1000.
            batch_size (int, optional): Batch Size. Defaults to 64.
            patience (int, optional): Number of epochs to wait without improvement. Defaults to 10.
        """
        if os.path.isfile(model_save_path):
            print(
                "A Model in .hdf5 format already exists in given path. Overwrite? (y/n)"
            )
            val = input()
            if val.lower() == "y":
                self.model.save(model_save_path, include_optimizer=False)
            else:
                raise Exception("Model already exists in given path, aborting...")

        if os.path.isfile(tflite_save_path):
            print(
                "A Model in .tflite format already exists in given path. Overwrite? (y/n)"
            )
            val = input()
            if val.lower() == "y":
                pass
            else:
                raise Exception("Model already exists in given path, aborting...")

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            verbose=1,
            save_weights_only=False,
            save_best_only=True,
        )
        # Callback for early stopping
        es_callback = tf.keras.callbacks.EarlyStopping(patience=patience, verbose=1)

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            validation_data=(val_data, val_labels),
            callbacks=[cp_callback, es_callback],
            batch_size=batch_size,
        )

        self.convert_to_tflite(tflite_save_path)

        logging.info(' Loading  TFLite model from "{}"'.format(tflite_save_path))
        self.load(tflite_save_path)
        self.trained = True

    def convert_to_tflite(self, tflite_save_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        logging.info(" Converting to TFLite")
        tflite_quantized_model = converter.convert()
        open(tflite_save_path, "wb").write(tflite_quantized_model)

    def predict(self, landmark_list):
        """Takes a list of mediapipe keypoints and returns the predicted class

        Args:
            landmark_list (np.ndarray): A list of mediapipe keypoints

        Raises:
            Exception: When no path was specified and model.train() was not called

        Returns:
            int: Result of gesture prediction
        """
        if self.model is not None and not self.trained:
            raise Exception(
                "No Model path was given, and model.train() was not invoked"
            )
        input_details_tensor_index = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_details_tensor_index, np.array([landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]["index"]
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))

        return result_index

    def save(self, model_save_path):
        if os.path.isfile(model_save_path):
            print("A Model already exists in given path. Overwrite? (y/n)")
            val = input()
            if val == ord("y"):
                self.model.save(model_save_path, include_optimizer=False)
                return True
        return False

    def summary(self):
        return self.model.summary()
