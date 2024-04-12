# Produce inputs and their corresponding targets
# Aim : To produce a data genertor for our model
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
import pickle
import os
from sklearn.utils import shuffle

# #_______________________________________________________________________
save_path = '../Data/'

with open(os.path.join(save_path, "train_data.pickle"), 'rb') as f:
    (train_data, train_range) = pickle.load(f)

with open(os.path.join(save_path, "val_data.pickle"), 'rb') as f:
    (val_data, val_range) = pickle.load(f)

rng = np.random.RandomState(seed=42)
# ____________________________________________________________________


def get_batch(batch_size, s="train"):
    """Create batch of n pairs, half same class, half different class"""
    if s == 'train':
        X = train_data
        categories = train_range
    else:
        X = val_data
        categories = val_range
    n_classes, n_examples, w, h = X.shape

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes, size=(batch_size,), replace=False)

    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets

# ________________________________________________________________________________________________-


def data_generator(batch_size, s="train"):
    while True:
        pairs, targets = get_batch(batch_size, s)
        yield pairs, targets


# Test Data Generation
# 1. function to generate pairs of test images and support set.
# 2. we fill array for test images to N copies of one class (N user defined)
# 3. support set will contain N classes, which will include the above class


def make_oneshot_task(N, s="val", language=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = train_data
        categories = train_range
    else:
        X = val_data
        categories = val_range
    n_classes, n_examples, w, h = X.shape

    indices = rng.randint(0, n_examples, size=(N,))
    if language is not None:  # if language is specified, select characters for that language
        low, high = categories[language]
        if N > high - low:
            raise ValueError(
                "This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low, high), size=(N,), replace=False)

    else:  # if no language specified just pick a bunch of random letters
        categories = rng.choice(range(n_classes), size=(N,), replace=False)
    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
    test_image = np.asarray(
        [X[true_category, ex1, :, :]]*N).reshape(N, w, h, 1)
    support_set = X[categories, indices, :, :]
    support_set[0, :, :] = X[true_category, ex2]
    support_set = support_set.reshape(N, w, h, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(
        targets, test_image, support_set)
    pairs = [test_image, support_set]

    return pairs, targets

# ________________________________________________________________)_______________________________

# #### Testing the model
# 1. now we use above function to get test data
# 2. predict this data with the model
# 3. print relevant information


def test_oneshot(model, N, k, s="val", verbose=0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N, s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
    return percent_correct
######################################################################################################################
