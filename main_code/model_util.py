import pickle

#  t = {}
#  t[((1, 2), 0)] = [4, 5]


def save_model(model, file_name):
    """
    Save model
    """
    with open(file_name, "wb") as pickle_out:
        pickle.dump(model, pickle_out)


def load_model(file_name):
    """
    Load model
    """
    with open(file_name, "rb") as pickle_in:
        model = pickle.load(pickle_in)
        return model
