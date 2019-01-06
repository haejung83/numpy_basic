import numpy as np
import pickle

from PIL import Image
from dataset.mnist import load_mnist
from activation import sigmoid
from identity import softmax

# Referenced "Deep Learning from Scratch"

cached_data = dict()


def show_mnist():
    (img_train, label_train), (img_test, label_test) = load_mnist(normalize=False, flatten=True)

    img = img_train[0]
    label = label_train[0]

    print('Train Img Count: {}'.format(len(img_train)))
    print('Test Img Count: {}'.format(len(img_test)))

    print('Img shape: {}'.format(img.shape))
    print('Label: {}'.format(label))

    img = img.reshape(28, 28) # 28x28 = 784
    print('Img re-shape: {}'.format(img.shape))

    # pil_img = Image.fromarray(np.uint8(img))
    # pil_img.show()


def _load_data():
    # With normalized img is preventing to occur overflow at sigmoid function
    (img_train, label_train), (img_test, label_test) = load_mnist(normalize=True, flatten=True)
    cached_data['train'] = (img_train, label_train)
    cached_data['test'] = (img_test, label_test)


def get_train_data():
    if not 'train' in cached_data:
        _load_data()

    return cached_data['train']


def get_test_data():
    if not 'test' in cached_data:
        _load_data()

    return cached_data['test']


def init_network():
    with open('pretrained/mnist_forward.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    z3 = sigmoid(a3)
    y = softmax(z3)

    return y


def forward():
    (img_test, label_test) = get_test_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(img_test)):
        y = predict(network, img_test[i])
        p = np.argmax(y)
        if p == label_test[i]:
            accuracy_cnt += 1

    print("Accuracy: {}".format(accuracy_cnt/len(img_test)))


def forward_batch():
    (img_test, label_test) = get_test_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(img_test), batch_size):
        x_batch = img_test[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, 1) # Rank 1
        accuracy_cnt += np.sum(p == label_test[i:i+batch_size])

    print("Accuracy (Batch): {}".format(accuracy_cnt/len(img_test)))


if __name__ == '__main__':
    print("Forward")

    # Test Mnist load module
    show_mnist()

    # Run forward network with pretrained weight
    forward()

    # Run forward batch
    forward_batch()

