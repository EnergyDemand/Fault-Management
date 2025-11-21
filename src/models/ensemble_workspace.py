from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

def ensemble_classifier(X_train, y_train):
    models = list()
    # Define the hyperparameters in a dictionary
    common_params = {
        'n_estimators': 100,
        'random_state': 42
    }
    models.append(('GradientBoostingClassifier', GradientBoostingClassifier(**common_params)))
    models.append(('AdaBoost', AdaBoostClassifier(**common_params)))
    #models.append(('RandomForestClassifier', RandomForestClassifier(**common_params)))
    #models.append(('ET', ExtraTreesClassifier(**common_params)))
    # Initialize the ensemble classifier
    ensemle_clf = VotingClassifier(estimators=models, voting='soft')
    # Train the ensembler classifier
    ensemle_clf.fit(X_train, y_train)
    # Return the ensemble classifier
    return ensemle_clf


def individual_classifiers(X_train, y_train):
    # Define the hyperparameters in a dictionary
    common_params = {
        'n_estimators': 100,
        'random_state': 42
    }
    # Define the models in a dictionary
    untrained_models = {
    "GB": GradientBoostingClassifier(**common_params),
    "AB": AdaBoostClassifier(**common_params),
    "RF": RandomForestClassifier(**common_params),
    "ET": ExtraTreesClassifier(**common_params)
    }
    # Define dictionary to store the trained models
    trained_models = {}
    # Train each individual classifier and store in train_models list
    for name, model in untrained_models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    # Return the list of the trained indiviudal classifiers
    return trained_models


def random_forest_clf(X_train, y_train):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    return rf_clf