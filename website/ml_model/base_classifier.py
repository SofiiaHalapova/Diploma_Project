import os
import random
import string

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Ground_Detection_App.website.ml_model.load_data import LoadDataset

matplotlib.use('TkAgg')


class BaseClassifier:

    def __init__(self, class_names):
        self.data = None
        self.labels = None
        self.class_names = class_names
        self.wavelength = None
        self.scaler = None
        self.pca = None
        self.class_colors = sns.color_palette(None, len(class_names))

    def load_data(self, path):
        ld = LoadDataset(self.class_names)
        self.wavelength, self.labels, self.data = ld.load_data(path)

    def preprocess_data(self, pca_components=None, test_size=0.3, random_state=42, scale=True):
        # min-max normalization [0, 1] assuming a 16-bit range
        data = self.data.astype(np.float32) / 65535.0  # pixel values range from 0 to 65535 in 16-bit HSI

        #  reshaping the data and reconstructing outputs
        rows, cols, bands = data.shape

        # flatten the data
        X = data.reshape((rows * cols, bands))
        y = self.labels.ravel()

        # Filter out unlabeled pixels. Remove background (unlabeled) pixels
        mask = y > 0

        X = X[mask]
        y = y[mask] - 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                            stratify=y)

        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        if pca_components is not None:
            self.pca = PCA(n_components=pca_components)
            X_train = self.pca.fit_transform(X_train)
            X_test = self.pca.transform(X_test)

        return X_train, X_test, y_train, y_test, rows, cols, mask

    @staticmethod
    def save_prediction_to_file(method, fig=None, metric_type=None):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_img_path = os.path.join(current_dir, '..', 'static', 'imgs')
        os.makedirs(static_img_path, exist_ok=True)

        characters = string.ascii_letters + string.digits
        random_string = "".join(random.choice(characters) for _ in range(5))

        if fig is not None:
            full_img_path = os.path.join(static_img_path, f'{metric_type}_{method}_{random_string}.png')
            fig.savefig(full_img_path)
            plt.close(fig)
        else:
            full_img_path = os.path.join(static_img_path, f'classes_prediction_{method}_{random_string}.png')
            plt.savefig(full_img_path)
            plt.close()

        if metric_type is not None:
            return f'imgs/{metric_type}_{method}_{random_string}.png'
        else:
            return f'imgs/classes_prediction_{method}_{random_string}.png'

    def plot_predictions_truth(self, y_pred_img, y_img):
        plt.figure(figsize=(15, 7))

        cmap = ListedColormap(self.class_colors)

        plt.subplot(1, 2, 1)
        plt.imshow(y_pred_img, cmap=cmap)
        plt.title('Predicted Classes')
        plt.axis('off')

        patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=self.class_colors[i],
                            label="{:d}: {}".format(i, self.class_names[i]))[0] for i in range(len(self.class_names))]

        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.subplot(1, 2, 2)
        plt.imshow(y_img, cmap=cmap)
        plt.title('Ground Truths')
        plt.axis('off')

        plt.tight_layout()

    def flattent_data(self, mask, clf, rows, cols, y_test, X_test, method):
        y_pred_test = clf.predict(X_test)

        fig, metric_type = self.plot_confusion_matrix(y_test, y_pred_test)

        self.save_prediction_to_file(method, fig, metric_type)

        X_full = self.data.astype(np.float32) / 65535.0
        X_full = X_full.reshape((-1, X_full.shape[-1]))
        X_full_masked = X_full[mask]

        if self.scaler:
            X_full_masked = self.scaler.transform(X_full_masked)

        if self.pca:
            X_full_masked = self.pca.transform(X_full_masked)

        y_pred_raw = clf.predict(X_full_masked)

        y_pred_full = np.zeros(rows * cols, dtype=np.int32)
        y_pred_full[mask] = y_pred_raw + 1

        y_pred_img = y_pred_full.reshape((rows, cols))

        y_img = self.labels

        self.plot_predictions_truth(y_pred_img, y_img)

        return self.save_prediction_to_file(method)

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        fig, ax = plt.subplots(figsize=(12, 12))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        plt.tight_layout()

        return fig, "confusion_matrix"

    def print_prediction_distribution(self, y_pred):
        counts = pd.Series(y_pred).value_counts().sort_index()
        print("Predicted pixel distribution per class:")
        for i, count in enumerate(counts):
            print(f"{self.class_names[i]:30} - {count} pixels")

    def compute_classification_results(self):
        raise NotImplementedError("Must be implemented by subclasses")
