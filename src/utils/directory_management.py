import os
from datetime import datetime

# def get_file_dir(data_dir_path, voting_method='soft'):
#     """Generates the file structure to save the resulting data.
#         data_dir_path: data directory path
#     Returns:
#         file_dir_path: file directory path
#     """
#     now = datetime.now()
#     dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
#     res_file_name = f"evaluation-results-{voting_method}-{dt_string}"
#     file_dir_path = os.path.join(data_dir_path, res_file_name)
#     if not os.path.isdir(file_dir_path):
#         os.mkdir(file_dir_path)
#     return file_dir_path

def get_file_dir(data_dir_path, voting_method='soft', n_estimators=150):
    """Generates the file structure to save the resulting data.
        data_dir_path: data directory path
        voting_method: voting method used in the ensemble
        n_estimators: number of estimators used in models
        model_short_names: short names of included models
    Returns:
        file_dir_path: file directory path
    """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    # Include voting method, estimators, and model short names in the directory name
    res_file_name = f"evaluation-results-{voting_method}-estimators-{n_estimators}-{dt_string}"
    file_dir_path = os.path.join(data_dir_path, res_file_name)
    if not os.path.isdir(file_dir_path):
        os.mkdir(file_dir_path)
    return file_dir_path


def get_result_dir(experiment_label, label):
    """Generates the file structure to store resulting data.
    
    Args:
        experiment_label: String of the resulting experiment label.
        label: String of the resulting element.

    Returns:
        data_dir_path: data directory path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir_path = os.path.abspath(os.path.join(script_dir, ".."))
    project_dir_path = os.path.abspath(os.path.join(project_dir_path, ".."))
    # project_dir_path = os.getcwd()
    # project_dir_path = r"D:\GPU Check\naeem"
    data_dir_path = os.path.join(project_dir_path, "result")
    experiment_dir_path = os.path.join(data_dir_path, experiment_label)
    if not os.path.isdir(experiment_dir_path):
        os.mkdir(experiment_dir_path)
    data_dir_path = os.path.join(experiment_dir_path, label)
    if not os.path.isdir(data_dir_path):
        os.mkdir(data_dir_path)
    return data_dir_path

def get_data_dir():
    """Generates the file structure to get data.
    
    Args:
    
    Returns:
        data_dir_path: data directory path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir_path = os.path.abspath(os.path.join(script_dir, ".."))
    project_dir_path = os.path.abspath(os.path.join(project_dir_path, ".."))
    print(script_dir)
    print(project_dir_path)
    # project_dir_path = os.getcwd()
    # project_dir_path = r"D:\GPU Check\naeem"
    data_dir_path = os.path.join(project_dir_path, "data")
    if not os.path.isdir(data_dir_path):
        os.mkdir(data_dir_path)
    return data_dir_path
