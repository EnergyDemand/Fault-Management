from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from src.models.custom_voting_classifier import CustomVotingClassifier

def ensemble_classifier(X_train, y_train, n_estimators, voting_method = 'soft'):
    models = list()
    # Define the hyperparameters in a dictionary
    common_params = {
        'n_estimators': n_estimators,
        'random_state': 42
    }
    models.append(('GradientBoostingClassifier', GradientBoostingClassifier(**common_params)))
    models.append(('AdaBoost', AdaBoostClassifier(**common_params)))
    models.append(('CatBoost', CatBoostClassifier(**common_params, verbose=False)))
    models.append(('XGBoost', XGBClassifier(**common_params)))
    models.append(('LightGBM', LGBMClassifier(**common_params)))
    # Initialize the ensemble classifier
    if voting_method=='soft':
        ensemle_clf = VotingClassifier(estimators=models, voting=voting_method)
    elif voting_method=='hard':
        ensemle_clf = CustomVotingClassifier(estimators=models, voting=voting_method)
    # Train the ensembler classifier
    ensemle_clf.fit(X_train, y_train)
    # Return the ensemble classifier
    return ensemle_clf

# def ensemble_classifier(X_train, y_train, n_estimators, voting_method='soft'):
#     models = []
#     included_models = []  # To keep track of the included models' short names
#
#     # Define the hyperparameters in a dictionary
#     common_params = {
#         'n_estimators': n_estimators,
#         'random_state': 42
#     }
#
#     # Add Gradient Boosting to the ensemble and record its short name
#     models.append(('GradientBoostingClassifier', GradientBoostingClassifier(**common_params)))
#     included_models.append('GB')
#
#     models.append(('AdaBoost', AdaBoostClassifier(**common_params)))
#     included_models.append('AB')
#
#     models.append(('CatBoost', CatBoostClassifier(**common_params, verbose=False)))
#     included_models.append('CB')
#
#     models.append(('XGBoost', XGBClassifier(**common_params)))
#     included_models.append('XGB')
#
#     models.append(('LightGBM', LGBMClassifier(**common_params)))
#     included_models.append('LGBM')
#
#     # Initialize the ensemble classifier with selected voting method
#     if voting_method == 'soft':
#         ensemble_clf = VotingClassifier(estimators=models, voting=voting_method)
#     elif voting_method == 'hard':
#         ensemble_clf = VotingClassifier(estimators=models, voting=voting_method)
#
#     # Train the ensemble classifier
#     ensemble_clf.fit(X_train, y_train)
#
#     # Return the ensemble classifier and the list of included model short names
#     return ensemble_clf, included_models

def individual_classifiers(X_train, y_train, n_estimators):
    # Define the hyperparameters in a dictionary
    common_params = {
        'n_estimators': n_estimators,
        'random_state': 42
    }
    # Define the models in a dictionary
    untrained_models = {
    "GB": GradientBoostingClassifier(**common_params),
    "AB": AdaBoostClassifier(**common_params),
    "CB": CatBoostClassifier(**common_params),
    "XGBoost": XGBClassifier(**common_params),
    "LGBM": LGBMClassifier(**common_params)
    }
    # Define dictionary to store the trained models
    trained_models = {}
    # Train each individual classifier and store in train_models list
    for name, model in untrained_models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    # Return the list of the trained indiviudal classifiers
    return trained_models


# def random_forest_clf(X_train, y_train):
#     rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf_clf.fit(X_train, y_train)
#     return rf_clf