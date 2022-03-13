"""
Author: Tzuchen Liu
Supervisor: Faras Brumand
Master Thesis
The entry point of the training process of the classifier.
Run this file in order to start training.
This script includes training random forest classifier and
 multi-layer perceptron (neural network) classifier.
"""

import datetime as dt
import os
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
from app.helper_func.config import ClassificationConfig
from app.helper_func.utils import Saver


def train_classifier(config: ClassificationConfig):
    """
    The configurations of classifiers can be modified in
    class ClassificationConfig in app/helper_func/config.py
    """
    if config.print_on_screen:
        print(dt.datetime.now(), 'training starts')
    saver = Saver(config.log_dir)
    saver.backup_files(config.backup_file_list)
    all_dataset_list = os.listdir(config.dataset_path)
    all_dataset_list = sorted(all_dataset_list)
    for dataset_filename in all_dataset_list:
        if config.print_on_screen:
            print('processing file:', dataset_filename)
        dataset = pd.read_csv(config.dataset_path / dataset_filename)
        leakage_category = pd.cut(dataset.label, bins=config.category_cut_bins,
                                  labels=config.category_cut_labels)

        x_train, x_test, y_train, y_test = train_test_split(
            dataset.drop(columns=['label']),
            leakage_category,
            test_size=config.test_ratio)
        output_filename = dataset_filename.replace(
            config.input_file_extension, '')

        # Random Forest Classifier
        random_forest_classifier = RandomForestClassifier(
            n_estimators=config.n_estimators)
        random_forest_classifier.fit(x_train, y_train)

        y_pred = random_forest_classifier.predict(x_test)

        joblib.dump(
            random_forest_classifier,
            saver.log_dir_this_training /
            config.random_forest_classifier_name.replace(
                '*', output_filename),
            compress=config.compress)
        if config.print_on_screen:
            print(dt.datetime.now(), 'Random Forest Accuracy:',
                  metrics.accuracy_score(y_test, y_pred))

        # Neural Network Classifier (Multi-Layer Perceptron)
        neural_network_classifier = MLPClassifier(
            random_state=config.random_state, max_iter=config.max_iter)
        neural_network_classifier.fit(x_train, y_train)

        y_pred = neural_network_classifier.predict(x_test)

        joblib.dump(
            neural_network_classifier,
            saver.log_dir_this_training /
            config.neural_network_classifier_name.replace(
                '*', output_filename),
            compress=config.compress)
        if config.print_on_screen:
            print(dt.datetime.now(), 'Neural Network Accuracy:',
                  metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    CONFIGURATION = ClassificationConfig()
    train_classifier(CONFIGURATION)
