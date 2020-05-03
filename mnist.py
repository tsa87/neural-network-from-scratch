from layer import *
from functions import *
from network import *

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

initializer = "he"
momentum = 0.5
dropout_p = 0.25
batch_size = 32

network = []
network.append(Dense(X_train.shape[1], 200, initializer=initializer, momentum=momentum))
network.append(Leaky_ReLU())
network.append(Dense(200, 200, initializer=initializer, momentum=momentum, dropout_p=dropout_p))
network.append(Leaky_ReLU())
network.append(Dense(200, 10, initializer=initializer, momentum=momentum, dropout_p=dropout_p))

train_log = []
val_log = []

for epoch in range(25):
    for x_batch, y_batch in iterate_mini_batches(X_train, y_train, batch_size):
        weights = []
        train(network, x_batch, y_batch, weights=weights, l2_a=0.0005)

    train_log.append(np.mean(predict(network,X_train)==y_train))
    val_log.append(np.mean(predict(network,X_val)==y_val))

    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
    print("Val accuracy:",val_log[-1])
