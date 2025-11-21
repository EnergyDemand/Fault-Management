from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def normalize_data(df):
    """
    Normalize the data using min-max normalization method

    Args:
        df: Contains input data features.
    Returns:
        df: Normalized input data features within the range of [0,1] 
    """
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    for col in df.columns:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    return df


def get_data(df_data, label, data_split_ratio=0.3):
    """
    Set up the data

    Args:
        df_data: Contains input and output data features.
        label: String of the resulting element.
    Returns:
        X_train: training samples including input features 
        X_test: testing samples including input features
        y_train: label associated with training samples 
        y_test: label associated with testing samples
    """
    # Get and drop target label from the loaded dataframe
    target_class = df_data[label]
    df_data.drop(label, axis=1, inplace=True)
    features_names = df_data.columns.to_list()
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(df_data, target_class, test_size=data_split_ratio, random_state=1234)
    # Return the splited data
    return X_train, X_test, y_train, y_test, features_names


def new_get_data(df_data, label, data_split_ratio):

    # Split the data by class
    class_groups = df_data.groupby(label)

    # Determine the minimum number of samples available in each class (for equal sampling)
    min_class_samples = class_groups.size().min()
    # Print the minimum class samples
    print(f"Minimum class samples for test set: {min_class_samples}")
    min_class_samples = int(min_class_samples * data_split_ratio)


    # Create an empty DataFrame for test data
    test_data = pd.DataFrame()

    # Sample equal number of instances from each class for the test set
    for class_label, group in class_groups:
        test_data = pd.concat([test_data, group.sample(n=min_class_samples, random_state=42)])

    # Remove the test set samples from the original data to create the training set
    train_data = df_data.drop(test_data.index)

    y_train = train_data[label]
    train_data.drop(label, axis=1, inplace=True)

    y_test = test_data[label]
    test_data.drop(label, axis=1, inplace=True)
    # Return the splited data
    return train_data, test_data, y_train, y_test