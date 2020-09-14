import json
import numpy as np
import matplotlib.pyplot as plt

class Plots():
    def __init__(self):

        with open('config.json') as f:
            self.config = json.load(f)

    def rgb_plot(self,mean_rgb):
        if self.config['plot_show']=="True":
            f = np.arange(0, mean_rgb.shape[0])
            plt.plot(f, mean_rgb[:, 0], 'r', f, mean_rgb[:, 1], 'g', f, mean_rgb[:, 2], 'b')
            plt.title("Mean RGB - Complete")
            plt.show()

        else:
            print ('no plots allowed')


