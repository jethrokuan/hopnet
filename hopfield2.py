'''
Description: CS5340 - Hopfield Network
Name: Eugene Lim, Jethro Kuan
Matric No.: Your matric number, Your partner's matric number
'''

import torch
import matplotlib
import numpy as np
import glob

import matplotlib.pyplot as plt

from PIL import Image, ImageOps


matplotlib.use('Agg')


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
    img_size = np.prod(imgs[0].shape)
    weights = np.zeros((img_size, img_size))
    bias = np.zeros(img_size)
    for i in range(len(imgs)):
        img = np.reshape(imgs[i], (-1))
        weights += np.outer(img, img) / len(imgs)
    return weights, bias


def learn_maxpl(imgs, max_iter=1000, lr=0.01):
    img_size = np.prod(imgs[0].shape)
    # weights = fake_weights + fake_weights.T (to ensure symmetry)
    fake_weights = torch.zeros(img_size, img_size, requires_grad=True)
    bias = torch.zeros(img_size, 1, requires_grad=True)
    for i in range(max_iter):
        weights = fake_weights + torch.t(fake_weights)
        log_prob = torch.tensor(0)
        for j in range(len(imgs)):
            img = np.reshape(imgs[j], (-1, 1))
            img = torch.from_numpy(img).float()
            img_prob = torch.sigmoid(torch.mm(weights, img) + bias)
            log_prob = log_prob + torch.sum(torch.log(img_prob))
        log_prob.backward()
        if i % 100 == 0: print(log_prob)
        with torch.no_grad():
            fake_weights = fake_weights + lr * fake_weights.grad
            bias = bias + lr * bias.grad
        fake_weights.requires_grad = True
        bias.requires_grad = True
    fake_weights = fake_weights.detach().numpy()
    bias = bias.detach().numpy()
    return fake_weights + fake_weights.T, bias


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
    plt.show()


def compute_prob(x, pos, W, b):
    z = np.dot(W[pos], x) + b[pos]
    return 1 / (1 + np.exp(-z))


def recover(cimgs, W, b, max_iter=10):
    img_size = np.prod(cimgs[0].shape)
    rimgs = []
    for cimg in cimgs:
        x = np.copy(cimg)
        for itr in range(max_iter):
            prev_x = np.copy(x)
            for i in range(x.shape[0]):
                pos_prob = compute_prob(x, i, W, b)
                x[i] = 1 if pos_prob > 0.5 else -1
            if np.allclose(prev_x, x):
                print('Converged')
                break
        rimgs.append(x)
    return np.array(rimgs)


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

# Recover 2 -- Max Pseudo Likelihood
Wmpl, bmpl = learn_maxpl(imgs)
plt.imshow(Wmpl)
plt.show()
rimgs_mpl = recover(cimgs, Wmpl, bmpl)
np.save('mpl.npy', rimgs_mpl)
plot_results(imgs, cimgs, rimgs_mpl, "mpl.jpg")
