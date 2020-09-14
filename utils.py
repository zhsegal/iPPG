from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt


def draw(array):
    img=Image.fromarray(array,'RGB')
    plt.imshow(img)


