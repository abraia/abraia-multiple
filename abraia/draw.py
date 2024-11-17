import cv2
import math
import numpy as np

from .utils import hex_to_rgb


def draw_point(img, point, color, thickness = 2):
    center, radius = np.round(point).astype(np.int32), thickness
    cv2.circle(img, center, radius, color, -1)
    return img


def draw_line(img, line, color, thickness = 2):
    pt1, pt2 = line
    cv2.line(img, pt1, pt2, color, thickness)
    return img


def draw_ellipse(img, rect, color, thickness = 2):
    x, y, w, h =  np.round(rect).astype(np.int32)
    axes = round(w / 2), round(h / 2)
    center = round(x + w / 2), round(y + h / 2)
    cv2.ellipse(img, center, axes, 0, 0, 360, color, thickness)
    return img


def draw_filled_ellipse(img, rect, color):
    return draw_ellipse(img, rect, color, -1)


def draw_rectangle(img, rect, color, thickness = 2):
    x, y, w, h =  np.round(rect).astype(np.int32)
    pt1, pt2 = (x, y), (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, color, thickness)
    return img


def draw_filled_rectangle(img, rect, color):
    return draw_rectangle(img, rect, color, -1)


def draw_polygon(img, polygon, color, thickness = 2):
    points = np.round(polygon).astype(np.int32)
    cv2.polylines(img, [points], True, color, thickness)
    return img


def draw_filled_polygon(img, polygon, color, opacity = 1):
    points = np.round(polygon).astype(np.int32)
    cv2.fillPoly(img, [points], color)
    return img


def draw_blurred_mask(img, mask):
    w_k = int(0.1 * max(img.shape[:2]))
    w_k = w_k + 1 if w_k % 2 == 0 else w_k
    # m_k = int(math.sqrt(w_k))
    # m_k = m_k + 1 if m_k % 2 == 0 else m_k
    # print(w_k, m_k)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blurred_img = cv2.GaussianBlur(img, (w_k, w_k), 0)
    # blurred_mask = cv2.blur(mask, (5, 5))
    img = np.where(mask==0, img, blurred_img)
    # img1 = cv2.multiply(1 - (blurred_mask / 255), img)
    # img2 = cv2.multiply(blurred_mask / 255, blurred_img)
    # img = (cv2.add(img1, img2)).astype(np.uint8)
    return img


def draw_text(img, text, point, background_color = None, text_color = (255, 255, 255), 
              text_scale = 0.8, padding = 6):
    text_font, text_thickness = cv2.FONT_HERSHEY_DUPLEX, 1
    w, h = cv2.getTextSize(text, text_font, text_scale, text_thickness)[0]
    width, height = w + 2 * padding, h + 2 * padding
    x, y = point[0], max(point[1] - height, 0)
    org = (x + padding, y + padding + h)
    if background_color is not None:
        img = draw_filled_rectangle(img, [x, y, width, height], background_color)
    cv2.putText(img, text, org, text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)
    return img


def draw_overlay(img, overlay, rect = None, opacity = 1):
    x, y, width, height = rect if rect is not None else [0, 0, img.shape[1], img.shape[0]]
    x1, y1, x2, y2 = x, y, x + width, y + height
    overlay = cv2.resize(overlay, (width, height))
    alpha_channel = (overlay[:, :, 3] if overlay.shape[2] == 4 else np.ones((height, width), dtype=np.uint8) * 255)
    alpha_float = (cv2.convertScaleAbs(alpha_channel * opacity).astype(np.float32) / 255)[..., np.newaxis]
    blended_roi = cv2.convertScaleAbs((1 - alpha_float) * img[y1:y2, x1:x2] + alpha_float * overlay[:, :, :3])
    img[y1:y2, x1:x2] = blended_roi
    # img_copy = img.copy()
    # cv2.rectangle(img_copy, pt1, pt2, color, -1)
    # cv2.addWeighted(img_copy, opacity, img, 1 - opacity, 0, img)
    return img


def calculate_optimal_thickness(img_size):
    return 2 if min(img_size) < 1080 else 4


def calculate_optimal_text_scale(img_size):
    return max(min(img_size) * 0.0008, 0.8)


def render_results(img, results):
    thickness = calculate_optimal_thickness(img.shape[:2])
    text_scale = calculate_optimal_text_scale(img.shape[:2])
    for result in results:
        label = result.get('label')
        score = result.get('confidence')
        color = hex_to_rgb(result.get('color', '#009BFF'))
        if result.get('polygon'):
            draw_polygon(img, result['polygon'], color, thickness)
        elif result.get('box'):
            for point in result.get('keypoints', []):
                draw_point(img, point, color, thickness)
            draw_rectangle(img, result['box'], color, thickness)
        if (label):
            text = f"{label} {round(100 * score, 1)}%" if score else label
            point = result.get('box', [0, 0, 0, 0])[:2]
            draw_text(img, text, point, background_color=color, text_scale=text_scale, padding=thickness*3)
    return img
