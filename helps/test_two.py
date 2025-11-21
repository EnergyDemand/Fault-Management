import shap
import numpy as np
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import os

save_dir = './result'

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

# Usage with SHAP

# Assuming you've already set up the voting classifier `voting_clf`
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
    voting='hard'  # 'soft' for probability-based voting
)

# Fit the model
voting_clf.fit(X_train, y_train)



# Create the SHAP explainer using the custom wrapper
wrapped_voting_clf = VotingClassifierWrapper(voting_clf)
explainer = shap.Explainer(wrapped_voting_clf, X_train)
shap_values = explainer(X_test)

# Create directory to save SHAP plots
save_dir = "shap_plots_hard"
os.makedirs(save_dir, exist_ok=True)

# Example: Summary plot
shap.summary_plot(shap_values, X_test, feature_names=data.feature_names)
#shap.dependence_plot('sepal width (cm)', shap_values, X_test, feature_names=data.feature_names)

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=data.feature_names, show=False)
plt.savefig(os.path.join(save_dir, "summary_plot.png"))
plt.close()



# Example: Force plot for a single instance
X_sample = X_test[0:1]  # Select the first instance
shap_values_single = explainer(X_sample)

# # Save the waterfall plot for a single instance
# plt.figure()
# shap.plots.waterfall(shap_values_single[0], show=False)
# #shap.plots.waterfall(shap_values, show=False)
# plt.savefig(os.path.join(save_dir, "waterfall_plot.png"))
# plt.close()

# plt.figure()
# shap.waterfall_plot(shap_values[0])
# plt.savefig(os.path.join(save_dir, "waterfall_plot.png"))
# plt.close()

# Force plot for a single prediction
shap.initjs() 

# Save a summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=data.feature_names, show=False)
plt.savefig(os.path.join(save_dir, "summary_pplot.png"))
plt.close()

# Save a dependence plot (example using the first feature)
plt.figure()
shap.dependence_plot(0, shap_values.values, X_test, feature_names=data.feature_names, show=False)
plt.savefig(os.path.join(save_dir, "dependence_plot.png"))
plt.close()

# # Save the dependence plot
# plt.figure()
# shap.dependence_plot('sepal width (cm)', shap_values, X_test, feature_names=data.feature_names, show=False)
# plt.savefig(os.path.join(save_dir, "dependence_plot.png"))
# plt.close()
# Force plot for a single prediction
# shap.initjs()  # Initialize JS visualization for Jupyter
# shap.force_plot(explainer.expected_value[0], shap_values[0].values, X_test[0], feature_names=data.feature_names)