import cv2
import numpy as np

from .utils import hex_to_rgb


def load_image(src):
    return cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)


def save_image(dest, img):
    cv2.imwrite(dest, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def show_image(img):
    cv2.namedWindow('Image', cv2.WINDOW_GUI_NORMAL)
    cv2.imshow('Image',  cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyWindow('Image')


def draw_point(img, point, color, thickness = 2):
    center, radius = point, thickness
    cv2.circle(img, center, radius, color, thickness)
    return img


def draw_line(img, line, color, thickness = 2):
    pt1, pt2 = line
    cv2.line(img, pt1, pt2, color, thickness)
    return img


def draw_rectangle(img, rect, color, thickness = 2):
    x, y, w, h =  rect
    pt1, pt2 = (x, y), (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, color, thickness)
    return img


def draw_filled_rectangle(img, rect, color, opacity = 1):
    x, y, w, h = rect
    pt1, pt2 = (x, y), (x + w, y + h)
    if opacity == 1:
        cv2.rectangle(img, pt1, pt2, color, -1)
    else:
        img_copy = img.copy()
        cv2.rectangle(img_copy, pt1, pt2, color, -1)
        cv2.addWeighted(img_copy, opacity, img, 1 - opacity, 0, img)
    return img


def draw_polygon(img, polygon, color, thickness = 2):
    cv2.polylines(img, [polygon], True, color, thickness)
    return img


def draw_filled_polygon(img, polygon, color, opacity = 1):
    if opacity == 1:
        cv2.fillPoly(img, [polygon], color)
    else:
        img_copy = img.copy()
        cv2.fillPoly(img_copy, [polygon], color)
        cv2.addWeighted(img_copy, opacity, img, 1 - opacity, 0, img)
    return img


def draw_text(img, text, point, text_color = (255, 255, 255), text_scale = 0.5, text_thickness = 1, 
              text_font = cv2.FONT_HERSHEY_SIMPLEX, padding = 5, background_color = None):
    x, y = point
    w, h = cv2.getTextSize(text, text_font, text_scale, text_thickness)[0]
    rect = [x, max(y - h - 2 * padding, 0), w + 2 * padding, h + 2 * padding]
    org = (rect[0] + padding, rect[1] + padding + h)
    if background_color is not None:
        img = draw_filled_rectangle(img, rect, background_color)
    cv2.putText(img, text, org, text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)
    return img


def draw_overlay(img, overlay, rect, opacity = 1):
    x, y, width, height = rect
    x1, y1, x2, y2 = x, y, x + width, y + height
    overlay = cv2.resize(overlay, (width, height))
    alpha_channel = (overlay[:, :, 3] if overlay.shape[2] == 4 else np.ones((height, width), dtype=np.uint8) * 255)
    alpha_float = (cv2.convertScaleAbs(alpha_channel * opacity).astype(np.float32) / 255)[..., np.newaxis]
    blended_roi = cv2.convertScaleAbs((1 - alpha_float) * img[y1:y2, x1:x2] + alpha_float * overlay[:, :, :3])
    img[y1:y2, x1:x2] = blended_roi
    return img


def calculate_optimal_text_scale(img_size):
    return min(img_size) * 1e-3


def calculate_optimal_line_thickness(img_size):
    if min(img_size) < 1080:
        return 2
    return 4


def render_results(img, results):
    for result in results:
        label = result.get('label')
        prob = result.get('confidence')
        color = hex_to_rgb(result.get('color', '#009BFF'))
        x, y, w, h = result.get('box', [0, 0, 0, 0])
        if result.get('polygon'):
            draw_filled_polygon(img, result['polygon'], color, opacity=0.2)
            draw_polygon(img, result['polygon'], color, thickness=2)
        elif result.get('box'):
            if result.get('landmarks'):
                for point in result['landmarks'].values():
                    draw_point(img, point, color)
            else:
                # draw_filled_rectangle(img, result['box'], color, opacity=0.2)
                pass
            draw_rectangle(img, result['box'], color, thickness=2)
        if (label):
            text = f"{label} {round(100 * prob, 1)}%"
            draw_text(img, text, (x, y), background_color=color)
    return img
