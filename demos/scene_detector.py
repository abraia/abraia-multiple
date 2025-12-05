from scenedetect import detect, AdaptiveDetector
from abraia.utils import Video, show_image

video_src = '../images/cars.mp4'

frames = []
video = Video(video_src)
content_list = detect(video_src, AdaptiveDetector(), start_in_scene=True)
for content in content_list:
    frame_num = (content[1] - content[0]).frame_num // 2 + content[0].frame_num
    img = video.get_frame(frame_num)
    frames.append(img)


import cv2
import math
import numpy as np

row = math.ceil(math.sqrt(len(frames)))
col = math.ceil(len(frames) / row)
while len(frames) < row * col:
    frames.append(255 * np.ones_like(frames[0]))
scale = min(4000 / max(frames[0].shape[0] * row, frames[0].shape[1] * col), 1)
for k, frame in enumerate(frames):
    frames[k] = cv2.resize(frame, (0,0), fx=scale, fy=scale)

img = cv2.vconcat([cv2.hconcat(frames[r*col:(r+1)*col]) for r in range(row)])
show_image(img)