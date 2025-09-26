# This script generates train and test indices for your data, you must input number of samples (n_samples)
# For the toy data the n_samples is 50
import numpy as np
import joblib

def train_test_split_indices(n_samples, test_size=0.2, random_state=None):
    """
    Split indices into train and test sets.
    Parameters
    ----------
    n_samples : int
        Total number of samples in your dataset.
    test_size : float
        Fraction of data to use for testing (e.g. 0.2 = 20%).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    train_idx : np.ndarray
        Array of training indices.
    test_idx : np.ndarray
        Array of testing indices.
    """
    rng = np.random.default_rng(seed=random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_test = int(n_samples * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return train_idx, test_idx


def main():
    n_samples = 50 # for toy data num_samples is 50
    train_idx, test_idx = train_test_split_indices(n_samples, test_size=0.25, random_state=42)

    # Save the indices for later use
    train_test_indices = {'train_idx' : train_idx, 'test_idx': test_idx}
    joblib.dump(train_test_indices , 'train_test_indices.pkl')



if __name__ == "__main__":
    main()