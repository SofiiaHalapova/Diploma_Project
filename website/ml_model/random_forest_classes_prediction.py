import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

from Ground_Detection_App.website.ml_model.base_classifier import BaseClassifier

matplotlib.use('TkAgg')


class GeneratePredictionRandomForest(BaseClassifier):

    def __init__(self, class_names):
        super().__init__(class_names)

    def compute_classification_results(self):

        X_train, X_test, y_train, y_test, rows, cols, mask = self.preprocess_data(
            pca_components=None,
            scale=True,
            test_size=0.3,
            random_state=42
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # y_pred_full = np.zeros(mask.shape, dtype=np.int32)
        # X_full = self.data.reshape((-1, self.data.shape[-1])).astype(np.float32) / 65535.0
        #
        # if self.scaler:
        #     X_full = self.scaler.transform(X_full)
        # if self.pca:
        #     X_full = self.pca.transform(X_full)
        #
        # y_pred_full[mask] = clf.predict(X_full[mask])
        #
        # y_full = self.labels.ravel()
        #
        # y_img = y_full.reshape((rows, cols))
        # y_pred_img = y_pred_full.reshape((rows, cols))
        #
        # cmap = ListedColormap(self.class_colors)
        # self.plot_predictions_truth(cmap, y_pred_img, y_img, plt)
        #
        # return self.save_prediction_to_file(plt, 'random_forest')
        return self.flattent_data(mask, clf, rows, cols, 'random_forest')
