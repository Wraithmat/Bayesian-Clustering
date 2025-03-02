import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_ellipses(mean, covariance, ax):
    v, w = np.linalg.eigh(covariance)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(
        mean, v[0], v[1], angle=180 + angle
    )
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)
    ax.set_aspect("equal", "datalim")

if __name__=='__main__':
    gibbs = np.load('../data/not_hot.npy', allow_pickle=True).item()
    mus = gibbs['mus']
    Sigmas = gibbs['Sigmas']