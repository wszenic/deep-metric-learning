import glob

import matplotlib.pyplot as plt
import pytorch_lightning
import seaborn as sns
from PIL import Image


def save_emb_plots(out_path: str, network_module: pytorch_lightning.LightningModule):
    for iter, df in enumerate(network_module.res):
        sns.scatterplot(data=df, x="emb_0", y="emb_1", hue="labels")
        plt.title(f"Embeddings at iter = {iter}")
        plt.xlim((-4, 4))
        plt.ylim((-4, 4))
        plt.savefig(f"{out_path}/embeddings_at_iter_{iter:02d}.png")
        plt.clf()


def create_gif(in_path, out_path):
    res = []
    for file_name in sorted(glob.glob(f"{in_path}/*.png")):
        # Open each image file and append it to the list
        res.append(Image.open(file_name))

    # Save as an animated GIF
    res[0].save(f"{out_path}/learning_process.gif", save_all=True, append_images=res[1:], duration=300, loop=0)
