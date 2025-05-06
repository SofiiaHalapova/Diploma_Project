import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Ground_Detection_App.website.ml_model.load_data import LoadDataset


class BaseClassifier:

    def __init__(self, class_names):
        self.data = None
        self.method = None
        self.wavelength = None
        self.labels = None
        self.class_names = class_names
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        return X_train, X_test, y_train, y_test, rows, cols, mask

