import io
from PIL import Image
import matplotlib.pyplot as plt


class GifFactory:

    _fig_file: str = None
    images: list = None

    def __init__(self, fig_file):
        self._fig_file = fig_file
        self.img_buf = io.BytesIO()
        self.images = []

    def append_to_gif(self, fig: plt.Figure):
        img_buf = io.BytesIO()

        fig.savefig(img_buf, format="png")
        plt.close(fig)

        im = Image.open(img_buf)
        self.images.append(im.copy())

        img_buf.close()

    def save_gif(self):
        self.images[0].save(self._fig_file,
                            save_all=True,
                            append_images=self.images[1:],
                            duration=100,
                            loop=0)

