import numpy as np

from abraia.utils import draw


def test_render_status_uses_flush_black_background(monkeypatch):
    calls = []
    monkeypatch.setattr(
        draw,
        "draw_text",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    draw.render_status(image, fps=30, thickness=2)

    args, kwargs = calls[0]
    assert args[2] == (0, 100)
    assert kwargs["background_color"] == (0, 0, 0)
    assert kwargs["padding"] == 6


def test_render_resolution_uses_flush_black_background(monkeypatch):
    calls = []
    monkeypatch.setattr(
        draw,
        "draw_text",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    text_width = draw.cv2.getTextSize(
        "200x100", draw.cv2.FONT_HERSHEY_DUPLEX, 0.8, 1
    )[0][0]

    draw.render_resolution(image, text_scale=0.8, thickness=2)

    args, kwargs = calls[0]
    padding = kwargs["padding"]
    assert args[2] == (200 - text_width - 2 * padding, 100)
    assert kwargs["background_color"] == (0, 0, 0)
    assert padding == 6

