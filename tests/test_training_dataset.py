from abraia.training.dataset import Dataset
from unittest.mock import patch


@patch('abraia.training.dataset.list_datasets')
@patch('abraia.training.dataset.load_annotations')
@patch('abraia.training.dataset.list_images')
def test_dataset_load(mock_list_images, mock_load_annotations, mock_list_datasets):
    mock_list_datasets.return_value = ['test_project']
    mock_load_annotations.return_value = [{'filename': 'test.jpg', 'objects': [{'label': 'cat'}]}]
    mock_list_images.return_value = [{'name': 'test.jpg'}]
    
    ds = Dataset('test_project')
    ds.load()
    
    assert ds.project == 'test_project'
    assert ds.annotations == [{'filename': 'test.jpg', 'objects': [{ 'label': 'cat' }]}]
    assert ds.classes == ['cat']
    assert ds.task == 'classify'
    assert ds.images == [{'name': 'test.jpg'}]
    
    mock_list_datasets.assert_called_once()
    mock_load_annotations.assert_called_once_with('test_project')
    mock_list_images.assert_called_once_with('test_project')


@patch('abraia.training.dataset.save_annotations')
def test_dataset_save(mock_save_annotations):
    ds = Dataset('test_project')
    annotations = [{'filename': 'test.jpg', 'objects': []}]
    
    ds.save(annotations)
    
    assert ds.annotations == annotations
    mock_save_annotations.assert_called_once_with('test_project', annotations)


@patch('abraia.training.dataset.annotate_images')
def test_dataset_annotate_filter(mock_annotate_images):
    mock_annotate_images.return_value = [{'filename': 'new.jpg', 'objects': [{'label': 'cat'}]}]
    
    ds = Dataset('test_project')
    ds.images = [{'name': 'old.jpg'}, {'name': 'new.jpg'}]
    ds.annotations = [{'filename': 'old.jpg', 'objects': [{'label': 'dog'}]}]
    
    annotations = ds.annotate('cat')
    
    # Verify only new.jpg was passed to annotate_images
    mock_annotate_images.assert_called_once_with([{'name': 'new.jpg'}], ['cat'], segment=False)
    # Verify annotations are merged
    assert len(ds.annotations) == 2
    assert ds.annotations[0] == {'filename': 'old.jpg', 'objects': [{'label': 'dog'}]}
    assert ds.annotations[1] == {'filename': 'new.jpg', 'objects': [{'label': 'cat'}]}
