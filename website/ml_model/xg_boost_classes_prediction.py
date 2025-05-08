from xgboost import XGBClassifier

from Ground_Detection_App.website.ml_model.base_classifier import BaseClassifier


class GeneratePredictionXGBoost(BaseClassifier):

    def __init__(self, class_names):
        super().__init__(class_names)

    def compute_classification_results(self):
        X_train, X_test, y_train, y_test, rows, cols, mask = self.preprocess_data(
            pca_components=30,
            scale=True,
            test_size=0.3,
            random_state=42
        )

        clf = XGBClassifier(n_estimators=100, max_depth=5)
        clf.fit(X_train, y_train)

        return self.flattent_data(mask, clf, rows, cols, 'xg_boost')
