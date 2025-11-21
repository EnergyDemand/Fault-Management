import numpy as np
import pandas as pd
from src.utils.save_utils import loss_analysis, training_validation_analysis
from sklearn.metrics.pairwise import cosine_similarity

def calculate_reconstruction_loss(original, reconstructed):
    mse = np.mean(np.square(original - reconstructed), axis=1)
    return mse


def calculate_iqr_threshold(sample, q1=25, q2=75, c=1.5):
    # Calculate Q1 and Q3
    Q1 = np.percentile(sample, q1)
    Q3 = np.percentile(sample, q2)
    # Calculate the IQR
    IQR = Q3 - Q1
    # Determine the lower and upper thresholds
    lower_threshold = Q1 - c * IQR
    upper_threshold = Q3 + c * IQR
    print('Threshold:', str(lower_threshold))
    print('Threshold:', str(upper_threshold))
    return lower_threshold, upper_threshold


def calculate_cosine_similarity(X_train, reconstructed_array):
    X_train_array = X_train.to_numpy()
    ssim_values_train = np.zeros(len(X_train))
    data_range = X_train_array.max() - X_train_array.min()
    # Calculate SSIM for each sample in the training data
    for i in range(len(X_train)):
        # Reshape the arrays if necessary; for example, if your data is 1D
        # You might need to reshape into 2D or 3D depending on the SSIM input requirements
        original_sample = X_train_array[i].reshape(1, -1)  # Example reshape, adjust as needed
        r_sample = reconstructed_array[i].reshape(1, -1)
        try:
            cosine_sim = cosine_similarity(original_sample, r_sample)
            ssim_values_train[i] = cosine_sim[0,0]
        except ZeroDivisionError:
            ssim_values_train[i] = 0.0  # Handle division by zero by setting NaN
        # Determine the threshold based on the SSIM values from the training data
    #threshold_csim = np.percentile(ssim_values_train, 5)  # Example using the 5th per
    return ssim_values_train


def get_threshold(autoencoder, X_train_copy, data_dir, percentile_val=95):
    # Reconstruct samples using trained autoencoder
    reconstructed_samples = autoencoder.predict(X_train_copy)
    # Calculate reconstructed loss
    r_loss = calculate_reconstruction_loss(X_train_copy, reconstructed_samples)
    # Define a threshold for fault detection
    #threshold = np.percentile(r_loss, percentile_val) 
    #threshold = (np.sum(r_loss) /len(r_loss))
    #threshold = np.std(r_loss)
    # Calculate consine similarity between original and reconstructed samples
    ssim_values_train = calculate_cosine_similarity(X_train_copy.copy(), reconstructed_samples)
    # Calculate IQR based lower and upper threshold values using reconstruction loss
    lower_threshold, upper_threshold = calculate_iqr_threshold(r_loss)
    # Calculate IQR based lower and upper threshold values using cosine similarity
    lower_cs, upper_cs = calculate_iqr_threshold(ssim_values_train)
    # Store reconstructed loss and cosine similarity 
    r_loss_copy = r_loss.tolist()
    r_cs_copy = ssim_values_train.tolist()
    df_loss = pd.DataFrame()
    df_loss['trained_r_loss'] = r_loss_copy
    df_loss['trained_r_cs'] = r_cs_copy
    df_loss.to_csv(data_dir+'/trained_reconstruction_metrics.csv', index = False)
    # Return the resulting threshold values
    return lower_threshold, upper_threshold, lower_cs, upper_cs


def latent_features_processing(latent_representations, data_dir):
     # Train ensemble classifier using latent features space
    if isinstance(latent_representations, list) or isinstance(latent_representations, np.ndarray):
        # Convert the list to a DataFrame
        latent_representations = pd.DataFrame(latent_representations)
        # Optionally, you can add column names if you know them
        latent_representations.columns = [f'feature_{i}' for i in range(latent_representations.shape[1])]
        # Store features extracted from encoder
        latent_representations.to_csv(data_dir+'/latent_features_data.csv', index=False)
        # return pandas data frame of the latent features
        return latent_representations
    else:
        print(type(latent_representations))
        exit()
