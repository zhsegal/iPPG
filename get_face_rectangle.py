from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from IPython import display
from PIL import Image, ImageDraw, ImageOps
from utils import draw
import matplotlib.pyplot as plt

def get_rectangle(image,device):
    mtcnn = MTCNN(keep_all=True, device=device)
    image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    boxes, _ = mtcnn.detect(image)
    box=boxes[0]
    box=[round(abs(value)) for value in box]
    face=image[box[1]:box[3],box[0]:box[2]]
    # frame_draw = image.copy()
    # frame_draw=(Image.fromarray(frame_draw))
    # draw = ImageDraw.Draw(frame_draw)
    # for box in boxes:
    #     draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    return face