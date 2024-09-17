import cv2
import numpy as np

from PIL import Image


def load_video(src=0, callback=None, output=None):
    cap = cv2.VideoCapture(src)
    if cap.isOpened() == False:
        print("Error opening video file")
        return
    if output:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(w, h, fps, cap.isOpened())
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(w),int(h)))
    win_name = 'Video'
    cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if callback:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img = callback(img)
                frame = np.array(img)[:, :, ::-1].copy()
            if output:
                out.write(frame)
            cv2.imshow(win_name, frame)
            if (cv2.waitKey(25) & 0xFF == ord('q')) or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            # if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    if output:
        out.release()
    cv2.destroyWindow(win_name)
