import shap
import os
import matplotlib.pyplot as plt
from src.interpretation.voting_classifier import VotingClassifierWrapper


def ensemble_classifier_interpretation(ensemble_voting_clf, X_train, X_test, feature_names, voting_type, save_dir):
    # Create the SHAP explainer using the custom wrapper
    wrapped_voting_clf = VotingClassifierWrapper(ensemble_voting_clf)
    explainer = shap.Explainer(wrapped_voting_clf, X_train)
    shap_values = explainer(X_test)
    # Example: Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(save_dir, "summary_plot.png"))
    # plt.savefig(os.path.join(save_dir, "summary_plot.pdf"), dpi=900)
    plt.close()
    # Save a dependence plot (example using the first feature)
    plt.figure()
    shap.dependence_plot(0, shap_values.values, X_test, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(save_dir, "dependence_plot.png"))
    plt.close()
    
    if voting_type=='soft':
        # Example: Force plot for a single instance
        X_sample = X_test[0:1]  # Select the first instance
        shap_values_single = explainer(X_sample)
        # Save the waterfall plot for a single instance
        plt.figure()
        shap.plots.waterfall(shap_values_single[0], show=False)
        plt.savefig(os.path.join(save_dir, "waterfall_plot.png"))
        plt.close()

    