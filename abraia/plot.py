import math
import matplotlib.pyplot as plt


def plot_image(img, title=''):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def plot_images(imgs, titles=None, cmap='nipy_spectral'):
    plt.figure()
    k = len(imgs)
    r = int(math.sqrt(k))
    c = math.ceil(k / r)
    ax = plt.subplots(r, c)[1].reshape(-1)
    for i in range(k):
        if titles and len(titles) >= k:
            ax[i].title.set_text(titles[i])
        ax[i].imshow(imgs[i], cmap=cmap)
        ax[i].axis('off')
    plt.show()


def plot_train_history(history):
    plt.ylim(0, 1.01)
    plt.grid()
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training loss','Test accuracy'], loc='upper right')
