from abraia.utils import Video
from abraia.inference import PlateDetector
from abraia.editing import build_mask
from abraia.utils.draw import draw_blurred_mask

src = '1482016-hd_1920_1080_25fps.mp4'
src = '../images/cars.mp4'
video = Video(src)
detector = PlateDetector()

for k, frame in enumerate(video):
    plates = detector.detect(frame)
    mask = build_mask(frame, plates, [])
    out = draw_blurred_mask(frame, mask)
    video.show(out)
