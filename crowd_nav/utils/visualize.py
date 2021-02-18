import os
import torch
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_tsne(feat, index, output_dir):

    print("feat: global mean = {:.4f}, {:.4f}".format(feat.mean(), feat.std()))

    embed = TSNE(n_components=2).fit_transform(feat)

    cm_subsection = torch.true_divide(index, index.max())
    colors = cm.jet(cm_subsection)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(6, 4)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(embed[:, 0], embed[:, 1], c=colors, alpha=0.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.axis('tight')
    plt.savefig(os.path.join(output_dir, 'tsne.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_samples(primary, neighbor, goal, positive, negative, fname=None, window=3.0):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(6, 4)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(neighbor[:, 0]-primary[0], neighbor[:, 1]-primary[1], 'bo')
    ax.plot(goal[0]-primary[0], goal[1]-primary[1], 'k*')
    for i in range(neighbor.size(0)):
        ax.arrow(neighbor[i, 0]-primary[0], neighbor[i, 1]-primary[1], neighbor[i, 2]*0.1, neighbor[i, 3]*0.1, color='b', width=0.05)
    if len(positive.shape) < 2:
        positive = positive[None, :]
    ax.plot(positive[:, 0], positive[:, 1], 'gs')
    ax.plot(negative[:, 0], negative[:, 1], 'rx')
    ax.plot(0, 0, 'ko')
    ax.arrow(0, 0, primary[2]*0.1, primary[3]*0.1, color='k', width=0.05)
    ax.set_xlim(-window, window)
    ax.set_ylim(-window, window)
    ax.set_aspect('equal')
    plt.grid()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
