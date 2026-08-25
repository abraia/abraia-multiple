from tqdm import tqdm

from abraia import Abraia
from abraia.inference.clip import Clip
from abraia.inference.ops import search_vector
from abraia.utils import show_image


class ImageSearch:
    def __init__(self, project):
        self.project = project
        self.abraia = Abraia()
        self.clip_model = Clip()
        self.index = self._load_index()
        if self.index:
            print(f"Loaded index with {len(self.index)} images from {self.project}")

    def _load_index(self):
        try:
            return self.abraia.load_json(f"{self.project}/index.json")
        except:
            return []

    def create_index(self):
        folder_path = f"{self.project}/"
        files, _ = self.abraia.list_files(folder_path)
        files = [file for file in files if 'image' in file['type']]
        
        indexed_paths = {item['path'] for item in self.index}
        
        new_items = []
        print(f"Checking {len(files)} files for new images...")
        for file in tqdm(files):
            if file['path'] in indexed_paths:
                continue 
                
            try:
                img = self.abraia.load_image(file['path'])
                vector = self.clip_model.get_image_embeddings([img])[0]
                new_items.append({'path': file['path'], 'vector': vector.tolist()})
            except Exception as e:
                print(f"Error processing {file['path']}: {e}")
        
        if new_items:
            self.index.extend(new_items)
            self.abraia.save_json(f"{self.project}/index.json", self.index)
            print(f"Added {len(new_items)} new images to index.")
        else:
            print("No new images to index.")
            
        return self.index

    def search_similar(self, img, max_results=1):
        if not self.index:
            raise ValueError("Index is empty. Please call create_index() first.")
        
        query_vector = self.clip_model.get_image_embeddings([img])[0]
        idxs, scores = search_vector(query_vector, self.index, max_results)
        
        results = []
        for idx, score in zip(idxs, scores):
            result_path = self.index[idx]['path']
            print(f"Similar image: {result_path} (score: {score})")
            result_img = self.abraia.load_image(result_path)
            show_image(result_img)
            results.append({'path': result_path, 'score': score})
            
        return results

    def search_text(self, text, max_results=1):
        if not self.index:
            raise ValueError("Index is empty. Please call create_index() first.")
        
        query_vector = self.clip_model.get_text_embeddings([text])[0]
        idxs, scores = search_vector(query_vector, self.index, max_results)
        
        results = []
        for idx, score in zip(idxs, scores):
            result_path = self.index[idx]['path']
            print(f"Similar image: {result_path} (score: {score})")
            result_img = self.abraia.load_image(result_path)
            show_image(result_img)
            results.append({'path': result_path, 'score': score})
            
        return results
