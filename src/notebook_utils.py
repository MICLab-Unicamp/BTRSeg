'''
Utility functions related to doing stuff on Jupyter Notebooks
'''
import imageio
from matplotlib import pyplot as plt


def view_image(path, figsize=(15, 15)):
    visualize_preprocess = imageio.imread(path)
    fig = plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(visualize_preprocess, cmap="gray")
    plt.show()