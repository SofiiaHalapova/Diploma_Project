import os
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from matplotlib.colors import ListedColormap

from Ground_Detection_App.website.ml_model.base_classifier import BaseClassifier
from Ground_Detection_App.website.ml_model.load_data import LoadDataset

matplotlib.use('TkAgg')


class GeneratePredictionRandomForest(BaseClassifier):

    def __init__(self, class_names):
        super().__init__(class_names)
        # self.data = None
        self.method = None
        # self.labels = None
        # self.class_names = class_names
        # self.class_colors = sns.color_palette(None, len(class_names))

    # def preprocess_data(self):
    #     data = self.data.astype(np.float64) / 65535.0
    #
    #     rows, cols, bands = data.shape
    #
    #     X = data.reshape((rows * cols, bands))
    #     y = self.labels.ravel()
    #
    #     mask = y > 0
    #     X, y = X[mask], y[mask]
    #
    #     return X, y, rows, cols, mask

    # @staticmethod
    # def train_random_forest_classifier(X, y):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    #
    #     scaler = StandardScaler()
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.transform(X_test)
    #
    #     # Random Forest Classifier
    #     clf = RandomForestClassifier(n_estimators=100, max_depth=None, max_features="sqrt", class_weight="balanced",
    #                                  random_state=42, verbose=1)
    #     clf.fit(X_train_scaled, y_train)
    #
    #     return clf, scaler, X_test_scaled, y_test

    # def visualize_results(self, clf, scaler, X, rows, cols, mask, y_img):
    #     # Predict entire image
    #     X_scaled = scaler.transform(X)
    #     y_pred_full = clf.predict(X_scaled)
    #
    #     # Restore full image shape
    #     y_pred_img = np.zeros(rows * cols, dtype=int)
    #     y_pred_img[mask] = y_pred_full
    #     y_pred_img = y_pred_img.reshape((rows, cols))
    #     # class_colors = ['black', 'gray', 'green', 'brown', 'darkgreen',
    #     #                 'lightblue', 'tan', 'darkgray', 'red', 'purple']
    #     cmap = ListedColormap(self.class_colors)
    #
    #     plt.figure(figsize=(15, 7))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(y_pred_img, cmap=cmap)
    #     plt.title("Predicted Land Cover")
    #     plt.axis("off")
    #
    #     patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=self.class_colors[i],
    #                         label="{:d}: {}".format(i, self.class_names[i]))[0] for i in range(len(self.class_names))]
    #
    #     plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(y_img, cmap=cmap)
    #     plt.title('Ground Truth')
    #     plt.axis('off')
    #
    #     plt.tight_layout()
    #
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     static_img_path = os.path.join(current_dir, '..', 'static', 'imgs')
    #     os.makedirs(static_img_path, exist_ok=True)
    #
    #     full_img_path = os.path.join(static_img_path, 'classes_prediction_random_forest.png')
    #     plt.savefig(full_img_path)
    #     plt.close()
    #     return 'imgs/classes_prediction_random_forest.png'
    #
    # def compute_classification_results(self):
    #     X, y, rows, cols, mask = self.preprocess_data()
    #     clf, scaler, X_test, y_test = self.train_random_forest_classifier(X, y)
    #     y_img = np.zeros((rows, cols), dtype=int)
    #     y_img[mask.reshape((rows, cols))] = y
    #     return self.visualize_results(clf, scaler, X, rows, cols, mask, y_img)

    def compute_classification_results(self):

        X_train, X_test, y_train, y_test, rows, cols, mask = self.preprocess_data(
            pca_components=None,
            scale=True,
            test_size=0.3,
            random_state=42
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

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
        self.plot_predictions_truth(cmap, y_pred_img, y_img, plt)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_img_path = os.path.join(current_dir, '..', 'static', 'imgs')
        os.makedirs(static_img_path, exist_ok=True)
        full_img_path = os.path.join(static_img_path, 'classes_prediction_xg_boost.png')
        plt.savefig(full_img_path)
        plt.close()

        # return self.save_prediction_to_file(plt, 'random_forest')
        return f'imgs/classes_prediction_random_forest.png'

