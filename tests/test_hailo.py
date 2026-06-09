import numpy as np
from abraia.hailo.toolbox import VideoInput, VideoVisualizer

def test_video_input_init():
    input_data = VideoInput(
        input_src='images',
        batch_size=1
    )
    assert input_data.input_src == 'images'
    assert input_data.batch_size == 1
    assert input_data.input_type == 'images'

def test_video_visualizer_fps():
    visualizer = VideoVisualizer(
        save_output='test_output/output.png'
    )
    assert visualizer.save_output == 'test_output/output.png'
    
    visualizer.start()
    visualizer.increment(5)
    assert visualizer.count == 5
    summary = visualizer.frame_rate_summary()
    assert "Processed 5 frames" in summary

# def test_video_visualizer_save_output(tmp_path):
#     save_output = str(tmp_path / "output.png")
#     visualizer = VideoVisualizer(save_output=save_output)
    
#     frame = np.zeros((100, 100, 3), dtype=np.uint8)
#     visualizer.show(frame, fps=30.0, is_capture=False)
    
#     import os
#     assert os.path.exists(str(tmp_path / "output_0.png"))

# def test_video_visualizer_no_save_output(tmp_path):
#     visualizer = VideoVisualizer(save_output=None)
    
#     frame = np.zeros((100, 100, 3), dtype=np.uint8)
#     visualizer.show(frame, fps=30.0, is_capture=False)
    
#     import os
#     assert not os.path.exists(str(tmp_path / "output_0.png"))
