import pytest
import queue
import threading
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
        output_dir='test_output',
        save_output=True
    )
    assert visualizer.output_dir == 'test_output'
    assert visualizer.save_output is True
    
    visualizer.start()
    visualizer.increment(5)
    assert visualizer.count == 5
    summary = visualizer.frame_rate_summary()
    assert "Processed 5 frames" in summary

def test_video_input_visualizer_standalone():
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    video_input = VideoInput(input_src='images')
    video_visualizer = VideoVisualizer(save_output=False)
    
    # Preprocess (simulating the thread)
    video_input.preprocess(input_queue, model_input_width=640, model_input_height=640)
    
    # Check if something was put in the queue
    item = input_queue.get()
    assert item is not None
    raw_frames, processed_frames = item
    assert len(raw_frames) > 0
    
    # Mock inference result
    inference_result = [] 
    output_queue.put((raw_frames[0], inference_result))
    output_queue.put(None) # Sentinel
    
    def mock_callback(frame, result):
        return frame
    
    video_visualizer.visualize(output_queue, mock_callback, is_capture=False)
    
    assert video_visualizer.count == 1
