from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd

def smote_resampling(df_data, col):
    y_train = df_data[col]
    X_train = df_data.copy()
    X_train.drop([col], axis=1, inplace=True)
    # oversample = SMOTE(k_neighbors=1)
    oversample = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_resampled = oversample.fit_resample(X_train, y_train)

    # # Add a 'sample_type' column to differentiate original and synthetic
    # sample_type = ['original'] * len(df_data) + ['synthetic'] * (len(X_train_resampled) - len(df_data))
    # X_train_resampled[col] = y_resampled
    # X_train_resampled['sample_type'] = sample_type

    X_train_resampled[col] = y_resampled
    return X_train_resampled

def adasyn_resampling(df_data, col, random_state=42, n_neighbors=5):
    y_train = df_data[col]
    X_train = df_data.copy()
    X_train.drop([col], axis=1, inplace=True)
    adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
    X_train_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

    # # Add a 'sample_type' column to differentiate original and synthetic
    # sample_type = ['original'] * len(df_data) + ['synthetic'] * (len(X_train_resampled) - len(df_data))
    # X_train_resampled[col] = y_resampled
    # X_train_resampled['sample_type'] = sample_type

    X_train_resampled[col] = y_resampled
    return X_train_resampled