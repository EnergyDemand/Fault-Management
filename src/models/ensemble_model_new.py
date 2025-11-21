from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from src.models.custom_voting_classifier import CustomVotingClassifier

def ensemble_classifier(X_train, y_train, n_estimators, voting_method='soft'):
    # List of models to be included in the ensemble
    models = list()

    # Define the common hyperparameters for tree-based models
    common_params = {
        'n_estimators': n_estimators,
        'random_state': 4
    }

    # Define the hyperparameters for each individual model
    model_params = {
        'GradientBoosting': common_params,
        'AdaBoost': common_params,
        'CatBoost': common_params,
        'XGBoost': common_params,
        'LightGBM': common_params,
        'RandomForest': {'n_estimators': n_estimators, 'random_state': 42, 'max_depth': 10, 'verbose':1},
        # 'SVM': {'kernel': 'linear', 'C': 1.0, 'probability': True, 'random_state': 42,  'verbose':True}
        # 'KNN': {'n_neighbors': 5, 'weights': 'uniform'},
        # 'LogisticRegression': {'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
    }

    # Add individual models to the models list with specific parameters
    models.append(('GradientBoosting', GradientBoostingClassifier(**model_params['GradientBoosting'])))
    # models.append(('AdaBoost', AdaBoostClassifier(**model_params['AdaBoost'])))
    models.append(('CatBoost', CatBoostClassifier(**model_params['CatBoost'])))
    # models.append(('XGBoost', XGBClassifier(**model_params['XGBoost'])))
    # models.append(('LightGBM', LGBMClassifier(**model_params['LightGBM'])))
    models.append(('RandomForest', RandomForestClassifier(**model_params['RandomForest'])))
    # models.append(('SVM', SVC(**model_params['SVM'])))
    # models.append(('KNN', KNeighborsClassifier(**model_params['KNN'])))
    # models.append(('LogisticRegression', LogisticRegression(**model_params['LogisticRegression'])))

    # Initialize the ensemble classifier based on the voting method
    if voting_method == 'soft':
        ensemble_clf = VotingClassifier(estimators=models, voting=voting_method)
    elif voting_method == 'hard':
        ensemble_clf = CustomVotingClassifier(estimators=models, voting=voting_method)

    # Train the ensemble classifier
    ensemble_clf.fit(X_train, y_train)

    # Return the trained ensemble classifier
    return ensemble_clf

def individual_classifiers(X_train, y_train, n_estimators):
    # Define the common hyperparameters for tree-based models
    common_params = {
        'n_estimators': n_estimators,
        'random_state': 42
    }

    # Define the hyperparameters for each individual model
    model_params = {
        "GB": common_params,
        "AB": common_params,
        "CB": {'iterations': n_estimators, 'random_state': 42, 'verbose': False},
        "XGBoost": common_params,
        "LGBM": common_params,
        "RF": {'n_estimators': n_estimators, 'random_state': 42, 'max_depth': 10, 'verbose':1},
        # "SVM": {'kernel': 'linear', 'C': 1.0, 'probability': True, 'random_state': 42, 'verbose':True}
        # "KNN": {'n_neighbors': 5, 'weights': 'uniform'},
        # "LR": {'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
    }

    # Define a dictionary of untrained models
    untrained_models = {
        name: model_class(**params)
        for name, (model_class, params) in {
            "GB": (GradientBoostingClassifier, model_params["GB"]),
            # "AB": (AdaBoostClassifier, model_params["AB"]),
            "CB": (CatBoostClassifier, model_params["CB"]),
            "XGBoost": (XGBClassifier, model_params["XGBoost"]),
            "LGBM": (LGBMClassifier, model_params["LGBM"]),
            "RF": (RandomForestClassifier, model_params["RF"]),
            # "SVM": (SVC, model_params["SVM"]),
            # "KNN": (KNeighborsClassifier, model_params["KNN"]),
            # "LR": (LogisticRegression, model_params["LR"])
        }.items()
    }

    # Dictionary to store the trained models
    trained_models = {}

    # Train each individual classifier and store it in trained_models
    for name, model in untrained_models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    # Return the dictionary of trained individual classifiers
    return trained_models
