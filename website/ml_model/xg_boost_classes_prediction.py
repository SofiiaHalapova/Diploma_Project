import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from xgboost import XGBClassifier
import seaborn as sns

from Ground_Detection_App.website.ml_model.base_classifier import BaseClassifier
from Ground_Detection_App.website.ml_model.load_data import LoadDataset

matplotlib.use('TkAgg')


class GeneratePredictionXGBoost(BaseClassifier):

    def __init__(self, class_names):
        super().__init__(class_names)
        self.data = None
        self.method = None
        self.wavelength = None
        self.labels = None
        self.class_names = class_names
        self.class_colors = sns.color_palette(None, len(class_names))

    def load_data(self, path):
        ld = LoadDataset(self.class_names)
        self.wavelength, self.labels, self.data = ld.load_data(path)

    @staticmethod
    def train_xgboost(X_train, y_train, n_estimators=100, max_depth=5, min_child_weight=1, gamma=0.5, ):
        clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
        clf.fit(X_train, y_train)
        return clf

    def plot_predictions_truth(self, cmap, y_pred_img, y_img, class_colors):
        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        plt.imshow(y_pred_img, cmap=cmap)
        plt.title('Predicted Classes')
        plt.axis('off')

        patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=class_colors[i],
                            label="{:d}: {}".format(i, self.class_names[i]))[0] for i in range(len(self.class_names))]

        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.subplot(1, 2, 2)
        plt.imshow(y_img, cmap=cmap)
        plt.title('Ground Truths')
        plt.axis('off')

        plt.tight_layout()

    def compute_classification_results(self):
        X_train, X_test, y_train, y_test, rows, cols, mask = self.preprocess_data(
            pca_components=30,
            scale=True,
            test_size=0.3,
            random_state=42
        )

        clf = self.train_xgboost(X_train, y_train)

        # y_pred = clf.predict(X_test)

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
        self.plot_predictions_truth(cmap, y_pred_img, y_img, self.class_colors)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_img_path = os.path.join(current_dir, '..', 'static', 'imgs')
        os.makedirs(static_img_path, exist_ok=True)
        full_img_path = os.path.join(static_img_path, 'classes_prediction_xg_boost.png')
        plt.savefig(full_img_path)
        plt.close()

        return 'imgs/classes_prediction_xg_boost.png'
