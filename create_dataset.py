import os
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from models import KPClassifier
from dataset_collection import DataCollector


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_labels",
        type=int,
        default=1,
        help="Number of labels to collect data for",
    )
    parser.add_argument(
        "-s",
        "--samples_per_label",
        type=int,
        default=1250,
        help="Number of samples to collect for each label",
    )
    parser.add_argument(
        "-l",
        "--label_names",
        nargs="+",
        default=["label"],
        help="Names of the labels",
    )
    parser.add_argument(
        "-p",
        "--dataset_path",
        type=str,
        default="models/dataset.csv",
        help="Path to save the dataset",
    )
    return parser.parse_args()


def main():
    args = argparser()
    data_collector = DataCollector(
        num_labels=args.num_labels,
        samples_per_label=args.samples_per_label,
        label_names=args.label_names,
        dataset_path=args.dataset_path,
    )
    if os.path.isfile(args.dataset_path):
        val = input("A csv file already exists in given path. Overwrite? (y/n): ")
        if not val.lower() == "y":
            raise Exception("Dataset already exists in given path, aborting...")
        data_collector.collect()

    X_dataset = np.loadtxt(
        args.dataset_path,
        delimiter=",",
        dtype="float32",
        usecols=list(range(1, (21 * 2) + 1)),
    )
    y_dataset = np.loadtxt(args.dataset_path, delimiter=",", dtype="int32", usecols=(0))
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset,
        y_dataset,
        train_size=0.75,
        random_state=42,
    )

    model = KPClassifier(NUM_LABELS=args.num_labels)

    model.train(
        "models/keypoint_classifier.hdf5",
        "models/keypoint_classifier.tflite",
        X_train,
        y_train,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()
