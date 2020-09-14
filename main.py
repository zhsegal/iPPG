import torch
import numpy as np
import mmcv, cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from singal_process import rgb_mean, get_pulse_signal,get_heart_rate, get_mean_rgb
import json
from plots import Plots

def calculate_hr():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    video = mmcv.VideoReader('fast.mp4')
    framerate=video.fps
    window_size = int(framerate * 1.6)


    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video]

    mean_rgb=get_mean_rgb(frames, device=device, image_show=False)
    pulse_signal=get_pulse_signal(mean_rgb, window_size)
    heart_rate=get_heart_rate(pulse_signal,framerate=framerate)

    print (heart_rate)



if __name__ == '__main__':
    calculate_hr()

