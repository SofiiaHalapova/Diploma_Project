import os
import random
import string

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Ground_Detection_App.website.ml_model.load_data import LoadDataset


class BaseClassifier:

    def __init__(self, class_names):
        self.data = None
        # self.method = None
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
        # Normalize to [0, 1]
        # min-max normalization assuming a 16-bit range
        data = self.data.astype(np.float32) / 65535.0  # division by 65535 is typical for 16-bit hyperspectral images,
        # where pixel values range from 0 to 65535

        # rows,cols - spatial dimensions
        # bands - number of spectral features per pixel.
        #  reshaping the data and reconstructing outputs
        rows, cols, bands = data.shape
        print("rows: ", rows, " cols: ", cols, " bands: ", bands)

        # flatten the data
        # The 3D image of shape (rows, cols, bands) is reshaped into a
        # 2D matrix X of shape (num_pixels, bands)
        X = data.reshape(rows * cols, bands)

        # .ravel() flattens 2D data (rows, cols) into a 1D array of shape (rows * cols,)
        y = self.labels.ravel()

        # Filter out unlabeled pixels. Remove background (unlabeled) pixels
        mask = y > 0

        X = X[mask]
        y = y[mask]

        y = y - 1

        if scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        if pca_components is not None:
            self.pca = PCA(n_components=pca_components)
            X = self.pca.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                            stratify=y)

        return X_train, X_test, y_train, y_test, rows, cols, mask

    @staticmethod
    def save_prediction_to_file(method):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_img_path = os.path.join(current_dir, '..', 'static', 'imgs')
        os.makedirs(static_img_path, exist_ok=True)

        characters = string.ascii_letters + string.digits
        random_string = "".join(random.choice(characters) for _ in range(5))

        full_img_path = os.path.join(static_img_path, f'classes_prediction_{method}_{random_string}.png')
        plt.savefig(full_img_path)
        plt.close()

        return f'imgs/classes_prediction_{method}_{random_string}.png'

    def plot_predictions_truth(self, cmap, y_pred_img, y_img):
        plt.figure(figsize=(15, 7))

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

    def flattent_data(self, mask, clf, rows, cols, method):

        y_pred_full = np.zeros(mask.shape, dtype=np.int32)
        X_full = self.data.reshape((-1, self.data.shape[-1])).astype(np.float32) / 65535.0

        if self.scaler:
            X_full = self.scaler.transform(X_full)
        if self.pca:
            X_full = self.pca.transform(X_full)

        y_pred_full[mask] = clf.predict(X_full[mask])

        y_full = self.labels.ravel()

        y_img = y_full.reshape((rows, cols))
        y_pred_img = y_pred_full.reshape((rows, cols))

        cmap = ListedColormap(self.class_colors)
        self.plot_predictions_truth(cmap, y_pred_img, y_img)

        return self.save_prediction_to_file(method)

    def compute_classification_results(self):
        raise NotImplementedError("Must be implemented by subclasses")
