import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import argparse

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
    parser = argparse.ArgumentParser(description='Generate an animation of ellipses.')
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input data file (data.npy)')
    parser.add_argument('--input_gibbs', type=str, required=True, help='Path to the input gibbs file (not_hot.npy)')
    parser.add_argument('--output', type=str, required=True, help='Path to the output GIF file')
    args = parser.parse_args()

    p = np.load(args.input_data, allow_pickle=True)
    gibbs = np.load(args.input_gibbs, allow_pickle=True).item()
    mus = gibbs['mus']
    Sigmas = gibbs['Sigmas']

    frames=[]

    fig=plt.figure(figsize=(15,15))
    fig.canvas.draw()
    ax = fig.add_subplot(111)
    ax.set_xlim(min(p[:,0])-0.5, max(p[:,0])+0.5)
    ax.set_ylim(min(p[:,1])-0.5, max(p[:,1])+0.5)
    img = np.array(fig.canvas.renderer.buffer_rgba())  # Get image as array
    frames.append(Image.fromarray(img))  # Convert to PIL Image

    for i in range(0,len(mus)):
        ax.cla()
        ax.scatter(p[:, 0], p[:, 1], c=p[:, 2])
        for j in range(len(mus[i])):
            make_ellipses(mus[i][j], Sigmas[i][j], ax)
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(Image.fromarray(img))

    durations=[100]*int(len(frames)/10)+[50]*(len(frames)-int(len(frames)/10))
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
            