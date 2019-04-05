'''
Description: CS5340 - Hopfield Network
Name: Your Name, Your partner's name
Matric No.: Your matric number, Your partner's matric number
'''


import matplotlib
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import sgd
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def load_image(fname):
    img = Image.open(fname).resize((32, 32))
    img_gray = img.convert('L')
    img_eq = ImageOps.autocontrast(img_gray)
    img_eq = np.array(img_eq.getdata()).reshape((img_eq.size[1], -1))
    return img_eq


def binarize_image(img_eq):
    img_bin = np.copy(img_eq)
    img_bin[img_bin < 128] = -1
    img_bin[img_bin >= 128] = 1
    return img_bin


def add_corruption(img):
    img = img.reshape((32, 32))
    t = np.random.choice(3)
    if t == 0:
        i = np.random.randint(32)
        img[i:(i + 8)] = -1
    elif t == 1:
        i = np.random.randint(32)
        img[:, i:(i + 8)] = -1
    else:
        mask = np.sum([np.diag(-np.ones(32 - np.abs(i)), i)
                       for i in np.arange(-4, 5)], 0).astype(np.int)
        img[mask == -1] = -1
    return img.ravel()


def learn_hebbian(imgs):
    """Learn the weights and bias term for the Hopfield network with the Hebbian rule."""
    img_size = np.prod(imgs[0].shape)
    weights = np.zeros((img_size, img_size))
    bias = np.zeros(img_size)
    for img in imgs:
        img = np.reshape(img, -1)
        weights += np.outer(img, img)/ len(imgs)

    for diag in range(img_size):
        weights[diag][diag] = 0

    plt.imsave('weights_h.jpg', weights)

    return weights, bias


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def learn_maxpl(imgs):
    """Learn the weights and bias for the Hopfield network by maximizing the pseudo log-likelihood."""
    img_size = np.prod(imgs[0].shape)

    # Initialize the weights using Hebbian Rule
    fake_weights = np.random.normal(0, 0.1, (img_size, img_size))
    bias = np.random.normal(0, 0.1, (img_size))
    diag_mask = np.ones((img_size, img_size)) - np.identity(img_size)

    def objective(params, iter):
        fake_weights, bias = params
        weights = np.multiply((fake_weights + fake_weights.T) / 2, diag_mask)
        pll = 0
        for i in range(len(imgs)):
            img = np.reshape(imgs[i], -1)
            activations = np.matmul(weights, img) + bias
            output = sigmoid(activations)
            eps = 1e-10
            img[img < 0] = 0
            pll += np.sum(np.multiply(img, np.log(output+eps)) + np.multiply(1-img, np.log(1-output+eps)))
        if iter % 100 == 0: print(-pll)
        return -pll

    g = grad(objective)

    fake_weights, bias = sgd(g, (fake_weights, bias), num_iters=300, step_size=0.001)
    weights = np.multiply((fake_weights + fake_weights.T) / 2, diag_mask)

    plt.imsave('weights_mpl.jpg', weights)
    return weights, bias


def plot_results(imgs, cimgs, rimgs, fname='result.png'):
    '''
    This helper function can be used to visualize results.
    '''
    img_dim = 32
    assert imgs.shape[0] == cimgs.shape[0] == rimgs.shape[0]
    n_imgs = imgs.shape[0]
    fig, axn = plt.subplots(n_imgs, 3, figsize=[8, 8])
    for j in range(n_imgs):
        axn[j][0].axis('off')
        axn[j][0].imshow(imgs[j].reshape(img_dim, img_dim), cmap='Greys_r')
    axn[0, 0].set_title('True')
    for j in range(n_imgs):
        axn[j][1].axis('off')
        axn[j][1].imshow(cimgs[j].reshape(img_dim, img_dim), cmap='Greys_r')
    axn[0, 1].set_title('Corrupted')
    for j in range(n_imgs):
        axn[j][2].axis('off')
        axn[j][2].imshow(rimgs[j].reshape((img_dim, img_dim)), cmap='Greys_r')
    axn[0, 2].set_title('Recovered')
    fig.tight_layout()
    plt.savefig(fname)


def recover(cimgs, W, b):
    img_size = np.prod(cimgs[0].shape)
    ######################################################################
    ######################################################################
    rimgs = []
    for cimg in cimgs:
        x = np.copy(cimg)
        for itr in range(5):
            prev_x = np.copy(x)
            for i in range(x.shape[0]):
                v = np.matmul(W[i], cimg) + b[i]
                x[i] = 1 if v > 0 else -1
            if np.allclose(prev_x, x):
                break
        rimgs.append(x)
    rimgs = np.asarray(rimgs)
    # Complete this function
    # You are allowed to modify anything between these lines
    # Helper functions are allowed
    #######################################################################
    #######################################################################
    return rimgs


def main():
    # Load Images and Binarize
    ifiles = sorted(glob.glob('images/*'))
    timgs = [load_image(ifile) for ifile in ifiles]
    imgs = np.asarray([binarize_image(img) for img in timgs])

    # Add corruption
    cimgs = []
    for i, img in enumerate(imgs):
        cimgs.append(add_corruption(np.copy(imgs[i])))
    cimgs = np.asarray(cimgs)

    # Recover 1 -- Hebbian
    Wh, bh = learn_hebbian(imgs)
    rimgs_h = recover(cimgs, Wh, bh)
    np.save('hebbian.npy', rimgs_h)
    plot_results(imgs, cimgs, rimgs_h, "hebbian.jpg")

    # Recover 2 -- Max Pseudo Likelihood
    Wmpl, bmpl = learn_maxpl(imgs)
    rimgs_mpl = recover(cimgs, Wmpl, bmpl)
    np.save('mpl.npy', rimgs_mpl)
    plot_results(imgs, cimgs, rimgs_mpl, "mpl.jpg")


if __name__ == '__main__':
    main()
