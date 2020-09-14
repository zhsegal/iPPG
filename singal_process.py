import numpy as np
from plots import Plots
from scipy.signal import welch
import matplotlib.pyplot as plt
import mmcv, cv2
from get_face_rectangle import get_rectangle
from get_skin import get_skin_image
from utils import draw


def rgb_mean(masked_face,number_of_skin_pixels):
    r = np.sum(masked_face[:, :, 2]) / number_of_skin_pixels
    g = np.sum(masked_face[:, :, 1]) / number_of_skin_pixels
    b = np.sum(masked_face[:, :, 0]) / number_of_skin_pixels

    return r,g,b


def temp_norm(mean_rgb):
    mean_color = np.mean(mean_rgb, axis=1)
    diag_mean_color = np.diag(mean_color)
    diag_mean_color_inv = np.linalg.inv(diag_mean_color)
    temp_normed_rgb = np.matmul(diag_mean_color_inv, mean_rgb)

    return temp_normed_rgb

def get_pulse_signal(mean_rgb, window_size):
    pulse_signal = np.zeros(mean_rgb.shape[0])

    for window_start in range(0, (mean_rgb.shape[0] - window_size)):
        window_mean_rgb = mean_rgb[window_start:window_start + window_size - 1, :].T
        Plots().rgb_plot(window_mean_rgb.T)
        temp_normed_rgb = temp_norm(window_mean_rgb)
        Plots().rgb_plot(temp_normed_rgb.T)
        projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
        POS = np.matmul(projection_matrix, temp_normed_rgb)
        std = np.array([1, np.std(POS[0, :]) / np.std(POS[1, :])])
        pulse = np.matmul(std, POS)
        pulse_signal[window_start:window_start + window_size - 1] = pulse_signal[
                                                                    window_start:window_start + window_size - 1] + (
                                                                                pulse - np.mean(pulse)) / np.std(pulse)
    return pulse_signal


def get_heart_rate(signal,framerate, num_semgents=12):
    segment_length = (2*signal.shape[0]) // (num_semgents + 1)
    signal = signal.flatten()
    green_f, green_psd = welch(signal, framerate, 'flattop', nperseg=segment_length) #, scaling='spectrum',nfft=2048)
    first = np.where(green_f > 0.9)[0]  # 0.8 for 300 frames
    last = np.where(green_f < 1.8)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)
    max_idx = np.argmax(green_psd[range_of_interest])
    f_max = green_f[range_of_interest[max_idx]]

    hr = f_max * 60.0
    return hr

def get_mean_rgb(frames, device,image_show=False):
    mean_rgb = np.empty((0, 3))
    for i, frame in enumerate(frames):
        print(f'Tracking frame: {i + 1}')
        face = get_rectangle(frame, device=device)
        masked_face, skinMask = get_skin_image(face)
        r, g, b = rgb_mean(masked_face, np.sum(skinMask > 0))
        mean_rgb = np.append(mean_rgb, np.array([[r, g, b]]), axis=0)
        print(f"Mean RGB -> <R = {r}, G = {g}, B = {b} ")
        if image_show == True:
            cv2.imshow("video", masked_face)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    return mean_rgb