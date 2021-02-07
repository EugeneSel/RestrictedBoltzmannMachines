import numpy as np
from matplotlib import pyplot as plt

def read_alpha_digits(data, indices):
    dataset = []
    
    for idx in indices:
        dataset += [image.reshape(-1) for image in data[idx]]

    return np.array(dataset)


def plot_avg_mse(model_avg_mse, label, parameters):
    plot = plt.figure(figsize=(15, 10))

    for idx, avg_mse in enumerate(model_avg_mse):
        plt.plot(np.arange(len(avg_mse)), avg_mse, label=f"${label} = {parameters[idx]}$")

    plt.title(f'RBM training MSE retrospective for different amounts of ${label}$', fontsize=15)
    plt.xlabel('Number of epochs of RBM', fontsize=14)
    plt.ylabel('Average reconstruction error', fontsize=14)
    plt.legend(fontsize=12)

    plot.show()


def generate_image_per_rbm(rbms, label, parameters, iter_gibbs, nb_imgs_per_row=4):
    nb_rows = len(rbms) // nb_imgs_per_row + 1 \
            if len(rbms) % nb_imgs_per_row \
            else len(rbms) // nb_imgs_per_row
    
    plt.figure(figsize=(5 * nb_imgs_per_row, 6 * nb_rows))
    
    for idx, rbm in enumerate(rbms):
        test_img = rbm.generate_images(iter_gibbs=iter_gibbs, nb_images=1)[0]

        plt.subplot(nb_rows, nb_imgs_per_row, idx + 1)
        plt.title(f"${label} = {parameters[idx]}$")
        plt.imshow(test_img)

    plt.show()
