import random

import numpy as np
from keras import Sequential
from keras.src.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from scipy.ndimage import interpolation
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from Ground_Detection_App.website.ml_model.base_classifier import BaseClassifier


class GeneratePredictionCNN(BaseClassifier):

    def __init__(self, class_names):
        super().__init__(class_names)
        self.numComponents = 30
        self.patch_size = 7
        self.test_size = 0.25
        # self.class_names = [cls for cls in class_names if cls != 'background']

    @staticmethod
    def applyPCA(X, numComponents=75):
        # reducing dimensionality
        # newX.shape = (H * W, B)
        newX = X.reshape((-1, X.shape[-1]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)

        # reshaping back to 3D array
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

        return newX, pca

    @staticmethod
    def padWithZeros(X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    def createPatches(self, X, removeZeroLabels=True):
        y = self.labels

        margin = int((self.patch_size - 1) / 2)

        zeroPaddedX = self.padWithZeros(X, margin=margin)

        patchesX = np.zeros((X.shape[0] * X.shape[1], self.patch_size, self.patch_size, X.shape[2]))
        patchesY = np.zeros((X.shape[0] * X.shape[1]))

        patchIndex = 0

        for row in range(margin, zeroPaddedX.shape[0] - margin):
            for col in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[row - margin:row + margin + 1, col - margin:col + margin + 1]
                patchesX[patchIndex, :, :, :] = patch
                patchesY[patchIndex] = y[row - margin, col - margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            patchesX = patchesX[patchesY > 0, :, :, :]
            patchesY = patchesY[patchesY > 0]
            patchesY -= 1

        return patchesX, patchesY

    @staticmethod
    def AugmentData(X_train):
        flipped_patch = None
        for i in range(int(X_train.shape[0] / 2)):
            patch = X_train[i, :, :, :]
            num = random.randint(0, 2)
            if num == 0:
                flipped_patch = np.flipud(patch)
            if num == 1:
                flipped_patch = np.fliplr(patch)
            if num == 2:
                no = random.randrange(-180, 180, 30)
                flipped_patch = interpolation.rotate(patch, no, axes=(1, 0), reshape=False, output=None,
                                                     order=3, mode='constant', cval=0.0, prefilter=False)
            patch2 = flipped_patch
            X_train[i, :, :, :] = patch2
        return X_train

    def Patch(self, data, height_index, width_index):
        height_slice = slice(height_index, height_index + self.patch_size)
        width_slice = slice(width_index, width_index + self.patch_size)
        patch = data[height_slice, width_slice, :]

        return patch

    def train_model(self, X_train, y_train):

        model = Sequential()
        model.add(
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.patch_size, self.patch_size, self.numComponents)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(y_train.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

        model.summary()

        model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

        return model

    def generate_classified_image(self, X, model):

        height = self.labels.shape[0]
        width = self.labels.shape[1]

        print(height, width)

        patches = []
        positions = []
        outputs = np.zeros((height, width))

        margin = self.patch_size // 2
        for i in range(margin, height - margin):
            for j in range(margin, width - margin):
                target = int(self.labels[i, j])
                if target == 0:
                    continue
                image_patch = self.Patch(X, i - margin, j - margin)
                patches.append(
                    image_patch.reshape(1, self.patch_size, self.patch_size, self.numComponents).astype('float32'))
                positions.append((i, j))

        patches = np.concatenate(patches, axis=0)
        predictions = model.predict(patches)
        for prediction, position in zip(predictions, positions):
            outputs[position[0]][position[1]] = np.argmax(prediction) + 1

        return np.array(outputs.astype(int))

    def compute_classification_results(self):

        X, pca = self.applyPCA(self.data, numComponents=self.numComponents)
        XPatches, yPatches = self.createPatches(X)
        X_train, X_test, y_train, y_test = train_test_split(XPatches, yPatches, test_size=0.25,
                                                            random_state=345, stratify=yPatches)
        X_train = self.AugmentData(X_train)
        X_train = X_train.reshape(X_train.shape[0], -1)

        y_train = to_categorical(y_train)

        X_train = X_train.reshape(-1, self.patch_size, self.patch_size, self.numComponents)

        model = self.train_model(X_train, y_train)

        y_pred_img = self.generate_classified_image(X, model)

        self.plot_predictions_truth(y_pred_img, self.labels)

        return self.save_prediction_to_file("cnn"), y_pred_img


