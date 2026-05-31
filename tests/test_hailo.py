import pytest
import queue
import numpy as np
from abraia.hailo.toolbox import VideoPipeline, VideoInput, VideoVisualizer

def test_video_pipeline_init():
    pipeline = VideoPipeline(
        input_src='images',
        batch_size=1,
        output_dir='test_output',
        save_output=True,
        side_by_side=True
    )
    assert pipeline.input_src == 'images'
    assert pipeline.batch_size == 1
    assert pipeline.output_dir == 'test_output'
    assert pipeline.save_output is True
    assert pipeline.side_by_side is True
    assert pipeline.input_type == 'images'

def test_video_pipeline_fps():
    pipeline = VideoPipeline(input_src='images')
    pipeline.start()
    pipeline.increment(5)
    assert pipeline.count == 5
    summary = pipeline.frame_rate_summary()
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
