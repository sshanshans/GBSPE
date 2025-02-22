from src.utils.CoeffDict import CoeffDict
import pickle

def load_CoeffDict_from_pickle(path_to_file):
    with open(path_to_file, "rb") as f:
        T = pickle.load(f)
        return T