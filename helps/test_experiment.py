import shap
import numpy as np
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris



# Wrapper class for VotingClassifier
class VotingClassifierWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        # Get the predicted probabilities
        probs = self.model.predict_proba(X)
        
        # Handle multi-class or binary classification
        if len(probs[0].shape) > 1:
            # For multi-class classification, return all probabilities
            return np.array(probs)
        else:
            # For binary classification, return probability of the positive class
            return np.array([p[1] for p in probs])
# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the base classifiers
clf1 = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf2 = AdaBoostClassifier(n_estimators=100, random_state=42)
clf3 = ExtraTreesClassifier(n_estimators=100, random_state=42)

# Create the voting classifier
voting_clf = VotingClassifier(
    estimators=[('gb', clf1), ('ada', clf2), ('et', clf3)],
    voting='soft'  # 'soft' for probability-based voting
)

# Fit the model
voting_clf.fit(X_train, y_train)



# Create the SHAP explainer using the custom wrapper
wrapped_voting_clf = VotingClassifierWrapper(voting_clf)
explainer = shap.Explainer(wrapped_voting_clf, X_train)
shap_values = explainer(X_test)

# Example: Summary plot
shap.summary_plot(shap_values, X_test, feature_names=data.feature_names)