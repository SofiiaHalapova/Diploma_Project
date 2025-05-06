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
        self.method = None

    # def load_data(self, path):
    #     ld = LoadDataset(self.class_names)
    #     self.wavelength, self.labels, self.data = ld.load_data(path)

    @staticmethod
    def train_xgboost(X_train, y_train, n_estimators=100, max_depth=5, min_child_weight=1, gamma=0.5, ):
        clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
        clf.fit(X_train, y_train)
        return clf

    def compute_classification_results(self):
        X_train, X_test, y_train, y_test, rows, cols, mask = self.preprocess_data(
            pca_components=30,
            scale=True,
            test_size=0.3,
            random_state=42
        )

        clf = self.train_xgboost(X_train, y_train)

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
        self.plot_predictions_truth(cmap, y_pred_img, y_img, plt)

        # return self.save_prediction_to_file(plt, 'xg_boost')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_img_path = os.path.join(current_dir, '..', 'static', 'imgs')
        os.makedirs(static_img_path, exist_ok=True)
        full_img_path = os.path.join(static_img_path, 'classes_prediction_xg_boost.png')
        plt.savefig(full_img_path)
        plt.close()

        return f'imgs/classes_prediction_xg_boost.png'

