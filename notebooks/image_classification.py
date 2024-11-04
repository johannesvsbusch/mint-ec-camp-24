from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


def load_mnist():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Standardizing (255 is the total number of pixels an image can have)
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # One hot encoding the target class (labels)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def load_cifar10():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # Converting the pixels data to float type
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    # One hot encoding the target class (labels)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def visualize_mnist(images, labels=None, idxs=None, predictions = None):
    # draw 6 random images
    if idxs is None:
        idxs = np.random.choice(len(images), 6)
    # here we plot the images in 6 separate subfigures
    fig, axs = plt.subplots(2,3)
    fig.tight_layout()
    for i, ax in zip(idxs, axs.flatten()):
        ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        ax.imshow(images[i], cmap='gray', interpolation='none')
        title = f"True Label: {np.argmax(labels[i])}" + (f"\nPredicted: {np.argmax(predictions[i])}" if predictions is not None else str()) 
        ax.set_title(title)
            

def visualize_cifar10(images, labels, predictions=None):
    # Creating a list of all the class labels
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # draw 9 random images
    idxs = np.random.choice(len(images), 9)
    # here we plot the images in 6 separate subfigures
    fig, axs = plt.subplots(3,3)
    fig.tight_layout()
    for i, ax in zip(idxs,axs.flatten()):
        ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        ax.imshow(images[i], cmap=plt.cm.binary)
        ax.set_title(f"True Label: {class_names[np.argmax(labels[i])]}" +
        (f"\nPredicted: {class_names[np.argmax(predictions[i])]}" if predictions is not None else str()))


def plot_accuracies(accuracies):
    # this plots the accuracies in two plots for train and test set
    fig, [ax1, ax2] = plt.subplots(2,1)
    for accuracy in accuracies.values():
        ax1.plot(np.arange(1, len(accuracy["train"])+1), accuracy["train"])
        ax2.plot(np.arange(1, len(accuracy["train"])+1), accuracy["test"])
    fig.legend(accuracies.keys())
    ax2.set_xlabel("epoch")
    ax1.set_ylabel("train accuracy")
    ax2.set_ylabel("test accuracy")
    

def sweep_embedding_space(encoder, decoder, x_train):
    digit_size = 28
    scale = 1.0
    n = 30
    figure = np.zeros((digit_size * n, digit_size * n))
    
    z_train = encoder.predict(x_train)
    z_min = np.min(z_train, axis=0)
    z_max = np.max(z_train, axis=0)
    z1_space = np.linspace(z_min[0], z_max[0], n)
    z2_space = np.linspace(z_min[1], z_max[1], n)
    grid = np.meshgrid(z1_space, z2_space)
    z_grid = np.array((grid[0].ravel(), grid[1].ravel())).T
    x_grid = decoder.predict(z_grid)

    for i, yi in enumerate(z1_space):
            for j, xi in enumerate(z2_space):
                x_decoded = x_grid[i * j]
                digit = x_decoded.reshape(digit_size, digit_size)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

    figsize=15
    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(z1_space, 1)
    sample_range_y = np.round(z2_space, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z_0")
    plt.ylabel("z_1")
    plt.imshow(figure, cmap="Greys_r")
