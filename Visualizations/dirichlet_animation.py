import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import argparse
from learning import make_ellipses
from matplotlib.colors import ListedColormap

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate an animation of ellipses.')
    parser.add_argument('--input_data', type=str, default='./data/data_new.npy', help='Path to the input data file (data.npy)')
    parser.add_argument('--input_dirichlet', type=str, default='./data/dirproc.npy' , help='Path to the input gibbs file (not_hot.npy)')
    parser.add_argument('--output', type=str, default='./Visualizations/dir.gif', help='Path to the output GIF file')
    args = parser.parse_args()


    p = np.load(args.input_data, allow_pickle=True)
    dirproc = np.load(args.input_dirichlet, allow_pickle=True).item()    

    frames=[]

    fig=plt.figure(figsize=(15,15))
    fig.canvas.draw()
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=17)  # Increase tick font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(19)  # Increase title and label font size
    ax.set_xlim(min(p[:,0])-0.5, max(p[:,0])+0.5)
    ax.set_ylim(min(p[:,1])-0.5, max(p[:,1])+0.5)
    img = np.array(fig.canvas.renderer.buffer_rgba())  # Get image as array
    frames.append(Image.fromarray(img))  # Convert to PIL Image

    assert(len(np.unique(dirproc['assignements']))==int((len(dirproc.keys())-1)/2))
    n_seen = int((len(dirproc.keys())-1)/2)
    
    to_plot_number = np.zeros(n_seen, dtype=int)
    cmap = ListedColormap(plt.cm.viridis(np.linspace(1, 0, n_seen)))


    for i in range(0,600,4):
        ax.cla()
        scatter = plt.scatter(p[:, 0], p[:, 1], c=dirproc['assignements'][i],s=60, cmap=cmap, vmin=0, vmax=9)
        for j in np.unique(dirproc['assignements'][i]):
            make_ellipses(dirproc[f'mu_{j}'][to_plot_number[j]], dirproc[f'Sigma_{j}'][to_plot_number[j]], ax, facecolor=cmap(j))
            to_plot_number[j] += 1
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(Image.fromarray(img))

    durations=[100]*int(len(frames)/10)+[50]*(len(frames)-int(len(frames)/10))
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
            