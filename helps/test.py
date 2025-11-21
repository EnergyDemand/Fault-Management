import os
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier

save_dir = './result'

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual classifiers
clf1 = GradientBoostingClassifier()
clf2 = AdaBoostClassifier()
clf3 = ExtraTreesClassifier()

# Voting classifier (adjust voting to 'soft' or 'hard')
voting_clf = VotingClassifier(estimators=[('gb', clf1), ('ada', clf2), ('et', clf3)], voting='soft')

# Train the voting classifier
voting_clf.fit(X_train, y_train)

# Wrapper class for VotingClassifier to handle both soft and hard voting
class VotingClassifierWrapper:
    def __init__(self, model):
        self.model = model
        self.voting = model.voting  # 'soft' or 'hard'

    def __call__(self, X):
        if self.voting == 'soft':
            # Soft voting: return the predicted probabilities
            return np.array(self.model.predict_proba(X))
        elif self.voting == 'hard':
            # Hard voting: return binary 0/1 predictions as probabilities
            predictions = self.model.predict(X)
            # Convert predictions to a probability-like output (0 or 1)
            return np.expand_dims(predictions, axis=1)
        else:
            raise ValueError("Voting type must be either 'soft' or 'hard'.")

# Wrap the voting classifier
wrapped_voting_clf = VotingClassifierWrapper(voting_clf)

# KernelExplainer can be slow for large datasets, so we'll sample a small subset for the background data
background = shap.sample(X_train, 100)

# Create the SHAP explainer using the KernelExplainer (model-agnostic)
explainer = shap.KernelExplainer(wrapped_voting_clf, background)

# Obtain SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Create directory to save SHAP plots
save_dir = "shap_plots"
os.makedirs(save_dir, exist_ok=True)

# Example: Force plot for a single instance
X_sample = X_test[0:1]  # Select the first instance

# Save the force plot
shap.force_plot(explainer.expected_value, shap_values[0][0], X_sample, feature_names=data.feature_names).savefig(os.path.join(save_dir, "force_plot.png"))

# Save the force plot for multiple instances
shap.force_plot(explainer.expected_value, shap_values[0][:10], X_test[:10], feature_names=data.feature_names).savefig(os.path.join(save_dir, "force_plot_multiple.png"))

# Save the waterfall plot for a single instance
plt.figure()
shap.waterfall_plot(shap_values[0][0], X_sample[0], feature_names=data.feature_names)
plt.savefig(os.path.join(save_dir, "waterfall_plot.png"))
plt.close()

# Save a summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=data.feature_names, show=False)
plt.savefig(os.path.join(save_dir, "summary_plot.png"))
plt.close()

# Save a dependence plot (example using the first feature)
plt.figure()
shap.dependence_plot(0, shap_values[0], X_test, feature_names=data.feature_names, show=False)
plt.savefig(os.path.join(save_dir, "dependence_plot.png"))
plt.close()

print(f"SHAP plots saved in the directory: {save_dir}")
