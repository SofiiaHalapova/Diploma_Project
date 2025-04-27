import os

import numpy as np
import time

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import seaborn as sns


from Ground_Detection_App.website.ml_model.load_data import LoadDataset

matplotlib.use('TkAgg')


class GeneratePrediction:

    def __init__(self, class_names):
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
    def train_test_split(X, y, data_parts, n_iterations, i):
        train_indices = np.concatenate([data_parts[j] for j in range(n_iterations) if j != i])
        test_indices = data_parts[i]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        return train_indices, test_indices, X_train, X_test, y_train, y_test

    @staticmethod
    def preprocess_data_for_image(X_train, X_test, n_components=100):
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        pcX_train = pca.transform(X_train)
        pcX_test = pca.transform(X_test)
        return pcX_train, pcX_test, pca

    @staticmethod
    def train_xgboost(X_train, y_train, max_depth=5, n_estimators=100):
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
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        mask = y_true > 0

        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=list(self.class_names.keys())[1:])
        disp = ConfusionMatrixDisplay(cm, display_labels=list(self.class_names.values())[1:])

        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title("Confusion Matrix XGBoost")
        plt.show()

        print("\nClassification Report:\n",
              classification_report(y_true_filtered, y_pred_filtered, labels=list(self.class_names.keys())[1:],
                                    target_names=list(self.class_names.values())[1:], zero_division=0))

    def compute_classification_results(self, n_iterations=5):
        print('compute_classification_results is called')

        data = self.data.astype(np.float64) / 65535.0

        rows, cols, bands = data.shape
        y = self.labels.ravel()

        # спектральний вектор
        X = data.reshape((rows * cols, bands))
        y_pred_full = np.zeros_like(y)

        data_parts = np.array_split(np.random.permutation(len(X)), n_iterations)
        accuracy_scores = []

        for i in range(n_iterations):
            print("iteration {}".format(i + 1))

            train_indices, test_indices, X_train, X_test, y_train, y_test = self.train_test_split(X, y, data_parts,
                                                                                                  n_iterations, i)

            pcX_train, pcX_test, pca = self.preprocess_data_for_image(X_train, X_test)

            start = time.time()

            # XGBoost Classifier
            clf = self.train_xgboost(pcX_train, y_train)

            print("Час тренування:", time.time() - start)

            y_pred = clf.predict(pcX_test).reshape(-1, 1)
            y_pred_full[test_indices] = y_pred.ravel()

            accuracy_scores.append(accuracy_score(y_test, y_pred))

        print("Середня точність:", np.mean(accuracy_scores))

        y_pred_img = y_pred_full.reshape((rows, cols))
        y_img = y.reshape((rows, cols))

        cmap = ListedColormap(self.class_colors)

        self.plot_predictions_truth(cmap, y_pred_img, y_img, self.class_colors)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_img_path = os.path.join(current_dir, '..', 'static', 'imgs')
        os.makedirs(static_img_path, exist_ok=True)

        full_img_path = os.path.join(static_img_path, 'classes_prediction.png')
        plt.savefig(full_img_path)
        plt.close()
        return 'imgs/classes_prediction.png'

        # self.plot_confusion_matrix(y.ravel(), y_pred_full)


