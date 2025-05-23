import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from Ground_Detection_App.website.ml_model.base_classifier import BaseClassifier


class GeneratePredictionLogisticRegression(BaseClassifier):

    def __init__(self, class_names):
        super().__init__(class_names)

    def compute_classification_results(self):
        X_train, X_test, y_train, y_test, rows, cols, mask = self.preprocess_data(
            pca_components=None,
            scale=True,
            test_size=0.3,
            random_state=42
        )

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        present_labels = np.unique(y_test)
        target_names = [self.class_names[k] for k in present_labels]

        print(classification_report(y_test, clf.predict(X_test), target_names=target_names))

        return self.flattent_data(mask, clf, rows, cols, 'logistic-regression')
