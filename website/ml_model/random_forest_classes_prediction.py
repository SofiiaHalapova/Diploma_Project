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

from Ground_Detection_App.website.ml_model.load_data import LoadDataset

matplotlib.use('TkAgg')


class GeneratePredictionRandomForest:

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

    def plot_predictions_truth(self, cmap, y_pred_img, y_img):
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(y_pred_img, cmap=cmap)
        plt.title('Класифіковане зображення (Прогнозовані класи)')
        plt.axis('off')

        patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=self.class_colors[i],
                            label="{:d}: {}".format(i, self.class_names[i]))[0] for i in range(len(self.class_names))]

        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.subplot(1, 2, 2)
        plt.imshow(y_img, cmap=cmap)
        plt.title('Істинні класи')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def preprocess_data(self):
        data = self.data.astype(np.float64) / 65535.0

        rows, cols, bands = data.shape

        X = data.reshape((rows * cols, bands))
        y = self.labels.ravel()

        mask = y > 0
        X, y = X[mask], y[mask]

        return X, y, rows, cols, mask

    @staticmethod
    def train_random_forest_classifier(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, max_features="sqrt", class_weight="balanced",
                                     random_state=42, verbose=1)
        clf.fit(X_train_scaled, y_train)

        return clf, scaler, X_test_scaled, y_test


    def visualize_results(self, clf, scaler, X, rows, cols, mask, y_img):
        # Predict entire image
        X_scaled = scaler.transform(X)
        y_pred_full = clf.predict(X_scaled)

        # Restore full image shape
        y_pred_img = np.zeros(rows * cols, dtype=int)
        y_pred_img[mask] = y_pred_full
        y_pred_img = y_pred_img.reshape((rows, cols))
        # class_colors = ['black', 'gray', 'green', 'brown', 'darkgreen',
        #                 'lightblue', 'tan', 'darkgray', 'red', 'purple']
        cmap = ListedColormap(self.class_colors)

        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(y_pred_img, cmap=cmap)
        plt.title("Predicted Land Cover")
        plt.axis("off")

        patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=self.class_colors[i],
                            label="{:d}: {}".format(i, self.class_names[i]))[0] for i in range(len(self.class_names))]

        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.subplot(1, 2, 2)
        plt.imshow(y_img, cmap=cmap)
        plt.title('Ground Truth')
        plt.axis('off')

        plt.tight_layout()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_img_path = os.path.join(current_dir, '..', 'static', 'imgs')
        os.makedirs(static_img_path, exist_ok=True)

        full_img_path = os.path.join(static_img_path, 'classes_prediction_random_forest.png')
        plt.savefig(full_img_path)
        plt.close()
        return 'imgs/classes_prediction_random_forest.png'

    def compute_classification_results(self):
        X, y, rows, cols, mask = self.preprocess_data()
        clf, scaler, X_test, y_test = self.train_random_forest_classifier(X, y)
        y_img = np.zeros((rows, cols), dtype=int)
        y_img[mask.reshape((rows, cols))] = y
        return self.visualize_results(clf, scaler, X, rows, cols, mask, y_img)









    # def calculate_metrics(self, clf, X_test, y_test):
    #     y_pred = clf.predict(X_test)
    #
    #     accuracy = accuracy_score(y_test, y_pred)
    #     print(f"Загальна точність: {accuracy:.4f}")
    #
    #     cm = confusion_matrix(y_test, y_pred)
    #     class_names = list(self.CLASS_NAMES.values())[1:]  # Remove "Background"
    #
    #     # Plot confusion matrix
    #     plt.figure(figsize=(15, 7))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    #     plt.xlabel("Predicted Label")
    #     plt.ylabel("True Label")
    #     plt.title("Confusion Matrix Random Forest Classifier")
    #     plt.xticks(rotation=45)
    #     plt.show()
    #
    #     print("\nClassification Report:\n",
    #           classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
