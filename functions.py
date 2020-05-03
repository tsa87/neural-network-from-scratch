import numpy as np
import keras

def softmax_logits(predictions, ground_truth):
    assert True not in np.isnan(predictions)
    assert predictions.shape[0] == len(ground_truth)

    batch_size = len(ground_truth)
    z_correct = np.empty((batch_size, ))

    # for idx in batch_size:
    #     z_correct[idx] = predictions[idx, ground_truth[idx]]
    z_correct = predictions[np.arange(batch_size), ground_truth]

    loss = -z_correct + np.log(np.sum(np.exp(predictions), axis=-1))

    return loss


def grad_softmax_logits(predictions, ground_truth, l2=False):
    assert True not in np.isnan(predictions)

    ones_for_answers = np.zeros_like(predictions)
    ones_for_answers[np.arange(len(predictions)), ground_truth] = 1

    softmax = np.exp(predictions) / np.exp(predictions).sum(axis=-1,keepdims=True)

    return (- ones_for_answers + softmax) / predictions.shape[0]

#
# Training helper
#
def iterate_mini_batches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)

    size = len(inputs)

    indices = np.arange(size)
    if shuffle:
        indices = np.random.permutation(size)

    for start_idx in range(0, size - batch_size + 1, batch_size):
        excerpt = indices[start_idx: start_idx + batch_size]

        yield inputs[excerpt], targets[excerpt]

#
# MNIST Specific
#
def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test
