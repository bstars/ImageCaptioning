import sys
sys.path.append('..')

import numpy as np
from sklearn.manifold import TSNE
from scipy.io import loadmat
import matplotlib.pyplot as plt

from utils.data_utils import COCODataset

def tsne_plot(wordvec, idx_to_word, num=1000):
    tsne_model = TSNE(n_components=2, verbose=2)
    newvals = tsne_model.fit_transform(wordvec)

    plt.figure(figsize=(20,20))

    for i in np.random.randint(0, len(newvals),num):
        plt.scatter(newvals[i,0], newvals[i,1])
        plt.annotate(idx_to_word[i],
                     xy=(newvals[i,0], newvals[i,1]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title('t-SNE')
    plt.show()

if __name__ == "__main__":
    wordvec = np.array(loadmat('../embed_mat.mat')['embed'])

    coco_path = "./data"
    idx_to_word = COCODataset().idx_to_word


    tsne_plot(wordvec, idx_to_word, num=200)