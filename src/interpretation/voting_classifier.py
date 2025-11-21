import numpy as np


class VotingClassifierWrapper:
    def __init__(self, model):
        self.model = model
        self.voting = model.voting  # 'soft' or 'hard'

    def __call__(self, X):
        if self.voting == 'soft':
            # Soft voting: return the predicted probabilities
            probs = self.model.predict_proba(X)
             # Handle multi-class or binary classification
            if len(probs[0].shape) > 1:
                # For multi-class classification, return all probabilities
                return np.array(probs)
            else:
                # For binary classification, return probability of the positive class
                return np.array([p[1] for p in probs])
            #return np.array(self.model.predict_proba(X))
        elif self.voting == 'hard':
            # Hard voting: return binary 0/1 predictions as probabilities
            predictions = self.model.predict(X)
            # Convert predictions to a probability-like output (0 or 1)
            # We expand dims to simulate the probabilistic output for SHAP
            return np.expand_dims(predictions, axis=1)
        else:
            raise ValueError("Voting type must be either 'soft' or 'hard'.")

# # Wrapper class for VotingClassifier
# class VotingClassifierWrapper:
#     def __init__(self, model):
#         self.model = model

#     def __call__(self, X):
#         # Get the predicted probabilities
#         probs = self.model.predict_proba(X)
        
#         # Handle multi-class or binary classification
#         if len(probs[0].shape) > 1:
#             # For multi-class classification, return all probabilities
#             return np.array(probs)
#         else:
#             # For binary classification, return probability of the positive class
#             return np.array([p[1] for p in probs])