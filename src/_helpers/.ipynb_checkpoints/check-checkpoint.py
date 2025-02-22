from src._helpers.math import *
import pickle

def save_dict(dp, filename):
    # Save the dictionary using pickle
    with open(filename, 'wb') as file:
        pickle.dump(dp, file)
        
def load_dict_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def check_and_create_folder(folder_path):
    """
    Check if a folder exists, and if it does not, create the folder using pathlib.
    
    Args:
    folder_path (str): The path to the folder to check and potentially create.
    """
    path = Path(folder_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)  # parents=True allows creation of any missing parents
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


def log_message(message, file_id):
    if file_id:
        print(message, file=file_id)
    else:
        print(message)

def create_ylabel(which_rslt):
    labels = {
        'val_est': 'Estimate',
        'mul_err': 'Multiplicative Error',
        'add_err': 'Additive Error'
    }

    if which_rslt in labels:
        return labels[which_rslt]
    else:
        raise ValueError("Invalid value for 'which_rslt'")

def crete_ylim(which_rslt, gt, scale):
    if which_rslt=='val_est':
        return (gt-scale, gt+scale)
    elif which_rslt=='mul_err':
        return (0, scale)
    elif which_rslt=='add_err':
        return (0, scale)

