from Ground_Detection_App.website.ml_model.base_classifier import BaseClassifier
from sklearn.neighbors import KNeighborsClassifier


class GeneratePredictionKNearest(BaseClassifier):

    def __init__(self, class_names):
        super().__init__(class_names)

    def compute_classification_results(self):
        X_train, X_test, y_train, y_test, rows, cols, mask = self.preprocess_data(
            pca_components=None,
            scale=True,
            test_size=0.3,
            random_state=42
        )

        knn = KNeighborsClassifier(n_neighbors=len(self.class_names))

        knn.fit(X_train, y_train)

        return self.flattent_data(mask, knn, rows, cols, 'knn')
