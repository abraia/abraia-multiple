import cv2
import numpy as np


def get_color(idx):
    colors = ['#D0021B', '#F5A623', '#F8E71C', '#8B572A', '#7ED321',
              '#417505', '#BD10E0', '#9013FE', '#4A90E2', '#50E3C2', '#B8E986',
              '#000000', '#545454', '#737373', '#A6A6A6', '#D9D9D9', '#FFFFFF']
    return colors[idx % (len(colors) - 1)]


def hex_to_rgb(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


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


# def draw_filled_rectangle(img, rect, color):
#     return draw_rectangle(img, rect, color, -1)


def draw_filled_rectangle(img, rect, color, opacity = 1):
    x, y, w, h = np.round(rect).astype(np.int32)
    pt1, pt2 = (x, y), (x + w, y + h)
    if opacity == 1:
        cv2.rectangle(img, pt1, pt2, color, -1)
    else:
        img_copy = img.copy()
        cv2.rectangle(img_copy, pt1, pt2, color, -1)
        cv2.addWeighted(img_copy, opacity, img, 1 - opacity, 0, img)
    return img


def draw_polygon(img, polygon, color, thickness = 2):
    points = np.round(polygon).astype(np.int32)
    cv2.polylines(img, [points], True, color, thickness)
    return img


def draw_filled_polygon(img, polygon, color, opacity = 1):
    points = np.round(polygon).astype(np.int32)
    if opacity == 1:
        cv2.fillPoly(img, [points], color)
    else:
        img_copy = img.copy()
        cv2.fillPoly(img_copy, [points], color)
        cv2.addWeighted(img_copy, opacity, img, 1 - opacity, 0, img)
    return img


# def draw_blurred_mask(img, mask):
#     w_k = int(0.1 * max(img.shape[:2]))
#     w_k = w_k + 1 if w_k % 2 == 0 else w_k
#     blurred_img = cv2.GaussianBlur(img, (w_k, w_k), 0)
#     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#     img = np.where(mask==0, img, blurred_img)
#     return img


# def draw_blurred_polygon(img, polygon):
#     points = np.round(polygon).astype(np.int32)
#     mask = np.zeros(img.shape, dtype=np.uint8)
#     mask = cv2.fillPoly(mask, [points], 255)
#     return draw_blurred_mask(img, mask)


def draw_blurred_mask(img, mask):
    w_k = int(0.1 * max(img.shape[:2]))
    w_k = w_k + 1 if w_k % 2 == 0 else w_k
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blurred_img = cv2.GaussianBlur(img, (w_k, w_k), 0)
    img = np.where(mask==0, img, blurred_img)
    # img1 = cv2.multiply(1 - (blurred_mask / 255), img)
    # img2 = cv2.multiply(blurred_mask / 255, blurred_img)
    # img = (cv2.add(img1, img2)).astype(np.uint8)
    return img


def draw_overlay_mask(img, mask, color = (255, 0, 0), opacity = 1):
    img_copy = img.copy()
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    overlay[mask == 255] = color
    img_over = cv2.addWeighted(img_copy, 1 - opacity, overlay, opacity, 0)
    img_copy[mask == 255] = img_over[mask == 255]
    return img_copy


def calculate_contrast_text_color(background_color):
    r, g, b = background_color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


def draw_text(img, text, point, background_color = None, text_color = (255, 255, 255), 
              text_scale = 0.8, padding = 6):
    if text:
        text_font, text_thickness = cv2.FONT_HERSHEY_DUPLEX, 1
        w, h = cv2.getTextSize(text, text_font, text_scale, text_thickness)[0]
        width, height = w + 2 * padding, h + 2 * padding
        x, y = point[0], max(point[1] - height, 0)
        org = (x + padding, y + padding + h)
        if background_color is not None:
            img = draw_filled_rectangle(img, [x, y, width, height], background_color)
            text_color = calculate_contrast_text_color(background_color)
        cv2.putText(img, text, org, text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)
    return img


def draw_text_multiline(img, lines, point, background_color = None, text_color = (255, 255, 255), 
                        text_scale = 0.8, padding = 6):
    x, y = point
    for line in lines:
        draw_text(img, line, (x, y), background_color, text_color, text_scale, padding)
        text_font, text_thickness = cv2.FONT_HERSHEY_DUPLEX, 1
        h = cv2.getTextSize(line, text_font, text_scale, text_thickness)[0][1]
        y += h + 2 * padding + 2
    return img


def draw_overlay(img, overlay, rect = None, opacity = 1):
    x, y, width, height = rect if rect is not None else [0, 0, img.shape[1], img.shape[0]]
    x1, y1, x2, y2 = x, y, x + width, y + height
    overlay = cv2.resize(overlay, (width, height))
    alpha_channel = (overlay[:, :, 3] if overlay.shape[2] == 4 else np.ones((height, width), dtype=np.uint8) * 255)
    alpha_float = (cv2.convertScaleAbs(alpha_channel * opacity).astype(np.float32) / 255)[..., np.newaxis]
    blended_roi = cv2.convertScaleAbs((1 - alpha_float) * img[y1:y2, x1:x2] + alpha_float * overlay[:, :, :3])
    img[y1:y2, x1:x2] = blended_roi
    return img


def draw_mask(overlay, mask, box, color):
    x, y, w, h = map(int, box)
    mh, mw = mask.shape[:2]
    y1, y2 = max(0, y), min(overlay.shape[0], y + mh)
    x1, x2 = max(0, x), min(overlay.shape[1], x + mw)
    if y2 > y1 and x2 > x1:
        m_y1, m_y2 = y1 - y, y2 - y
        m_x1, m_x2 = x1 - x, x2 - x
        overlay[y1:y2, x1:x2][mask[m_y1:m_y2, m_x1:m_x2] == 1] = color


def calculate_optimal_thickness(img_size):
    return 2 if min(img_size) < 1080 else 4


def calculate_optimal_text_scale(img_size):
    return max(min(img_size) * 0.0008, 0.8)


# Joint pairs used for drawing pose estimations
JOINT_PAIRS = [
    [0, 1], [1, 3], [0, 2], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [12, 14], [13, 15], [14, 16]
]


def render_box(img, box, color, thickness=None):
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    draw_rectangle(img, box, color, thickness)


def render_mask(overlay, mask, box, color):
    x, y, w, h = map(int, box)
    mh, mw = mask.shape[:2]
    y1, y2 = max(0, y), min(overlay.shape[0], y + mh)
    x1, x2 = max(0, x), min(overlay.shape[1], x + mw)
    if y2 > y1 and x2 > x1:
        m_y1, m_y2 = y1 - y, y2 - y
        m_x1, m_x2 = x1 - x, x2 - x
        overlay[y1:y2, x1:x2][mask[m_y1:m_y2, m_x1:m_x2] == 1] = color


def render_polygon(img, polygon, color, thickness=None):
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    draw_polygon(img, polygon, color, thickness)


def render_label(img, label, point, color, score=None, track_id=None, text_scale=None, thickness=None):
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    text_scale = text_scale or calculate_optimal_text_scale(img.shape[:2])
    text = f"{label} {round(score, 2)}" if score is not None else label
    if track_id is not None:
        text = f"[{track_id}] {text}"
    draw_text(img, text, (int(point[0]), int(point[1])), background_color=color, text_scale=text_scale, padding=thickness * 3)


def render_trail(img, trail, color, thickness=None):
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    for i in range(1, len(trail)):
        draw_line(img, (trail[i - 1], trail[i]), color, thickness=thickness)
        draw_point(img, trail[i], color, thickness=thickness * 2)


def render_keypoints(img, keypoints, color, thickness=None):
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    for point in keypoints:
        draw_point(img, point, color, thickness)


def render_skeleton(img, keypoints, joint_scores, color, thickness=None, joint_threshold=0.5):
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    for joint0, joint1 in JOINT_PAIRS:
        if joint_scores[joint0] >= joint_threshold and joint_scores[joint1] >= joint_threshold:
            pt1 = (int(keypoints[joint0][0]), int(keypoints[joint0][1]))
            pt2 = (int(keypoints[joint1][0]), int(keypoints[joint1][1]))
            draw_line(img, (pt1, pt2), color, thickness=thickness)


def draw_masks_overlay(img, overlay, alpha=0.5):
    img_over = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    mask = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY) > 0
    img[mask] = img_over[mask]
    return img


def render_results(img, results, thickness=None, text_scale=None):
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    text_scale = text_scale or calculate_optimal_text_scale(img.shape[:2])
    overlay = np.zeros_like(img) if any('mask' in r for r in results) else None
    for result in results:
        label = result.get('label')
        score = result.get('score')
        track_id = result.get('track_id')
        class_id = result.get('class_id', 0)
        color = hex_to_rgb(result.get('color', get_color(track_id if track_id is not None else class_id)))
        if result.get('polygon'):
            render_polygon(img, result['polygon'], color, thickness)
        elif result.get('box'):
            box = result.get('box')
            render_keypoints(img, result.get('keypoints', []), color, thickness)
            if 'joint_scores' in result:
                render_skeleton(img, result['keypoints'], result['joint_scores'], (255, 0, 255), thickness)
            render_box(img, box, color, thickness)
            if 'mask' in result and overlay is not None:
                # result['mask'] is already cropped to the bounding box
                render_mask(overlay, result['mask'], result['box'], color)
        if label:
            point = result.get('box', [0, 0, 0, 0])[:2]
            render_label(img, label, point, color, score, track_id, text_scale, thickness)
        if 'trail' in result:
            render_trail(img, result['trail'], color, thickness)
    if overlay is not None:
        img = draw_masks_overlay(img, overlay, alpha=0.5)
    return img



def render_counter(img, line, text='', color=(0, 0, 255), thickness=None, text_scale=None):
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    text_scale = text_scale or calculate_optimal_text_scale(img.shape[:2])
    draw_line(img, line, color=color, thickness=thickness)
    draw_text(img, text, line[0], background_color=color, text_scale=text_scale)
    return img


def render_region(img, region, text='', color=(255, 0, 0), opacity=0.2, thickness=None, text_scale=None):
    point = np.min(region, axis=0).astype(np.int32)
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    text_scale = text_scale or calculate_optimal_text_scale(img.shape[:2])
    draw_polygon(img, region, color=color, thickness=thickness)
    draw_filled_polygon(img, region, color=color, opacity=opacity)
    draw_text(img, text, point, background_color=color, text_scale=text_scale)
    return img


def render_status(img, fps=None, thickness=None, text_scale=None):
    import psutil
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    text_scale = text_scale or calculate_optimal_text_scale(img.shape[:2])
    lines = []
    if fps is not None:
        lines.append(f"FPS: {round(fps, 1)}")
    lines.append(f"CPU: {psutil.cpu_percent()}%")
    lines.append(f"RAM: {round(psutil.virtual_memory().used / (1024**3), 2)} GB")
    return draw_text_multiline(img, lines, (10, 40), background_color=(192, 192, 192), text_scale=text_scale, padding=thickness*3)


def render_resolution(img, thickness=None, text_scale=None):
    height, width = img.shape[:2]
    text = f"{width}x{height}"
    thickness = thickness or calculate_optimal_thickness(img.shape[:2])
    text_scale = text_scale or calculate_optimal_text_scale(img.shape[:2])
    text_font, text_thickness = cv2.FONT_HERSHEY_DUPLEX, 1
    w, h = cv2.getTextSize(text, text_font, text_scale, text_thickness)[0]
    padding = thickness * 3
    point = (width - w - 2 * padding, height)
    return draw_text(img, text, point, background_color=(128, 128, 128), text_scale=text_scale, padding=padding)
