# import numpy as np
# from sklearn.ensemble import VotingClassifier

# # Custom Voting Classifier with Hard Voting
# class CustomVotingClassifier(VotingClassifier):
#     def _predict(self, X):
#         # Collect predictions from each estimator
#         predictions = []
#         for est in self.estimators_:
#             pred = est.predict(X)
#             # Ensure the predictions are 1D arrays
#             if pred.ndim > 1:
#                 pred = pred.ravel()  # Flatten to 1D if necessary
#             predictions.append(pred)
#
#         # Convert to a numpy array and transpose for shape (n_samples, n_estimators)
#         return np.asarray(predictions).T

import numpy as np
from sklearn.ensemble import VotingClassifier

# Custom Voting Classifier with Hard Voting
class CustomVotingClassifier(VotingClassifier):
    def _predict(self, X):
        # Collect predictions from each estimator (5 in your case)
        predictions = []
        for i, est in enumerate(self.estimators_):
            # Make prediction for each estimator and flatten the output if necessary
            pred = est.predict(X)
            pred = np.ravel(pred)  # Flatten the predictions to a 1D array
            predictions.append(pred)
            # Print predictions from each classifier
            print(f"Predictions from classifier {i + 1} ({est.__class__.__name__}): {pred}")

        # Stack predictions into shape (n_samples, n_classifiers)
        predictions = np.array(predictions).T  # Shape: (n_samples, n_classifiers)

        # Perform hard voting
        # Get majority vote for each instance across classifiers
        ensemble_prediction = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)

        # Print the final ensemble prediction
        print("Ensemble prediction (Majority Voting):", ensemble_prediction)

        return ensemble_prediction

    # Override the predict method to call _predict directly
    def predict(self, X):
        return self._predict(X)