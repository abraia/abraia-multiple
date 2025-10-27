from tqdm import tqdm
from glob import glob
from abraia.utils import load_image, show_image
from abraia.inference.clip import Clip
from abraia.inference.ops import search_vector

clip_model = Clip()

print("Building image index...")
image_paths = glob('../images/*.jpg')
image_index = [{'vector': clip_model.get_image_embeddings([load_image(image_path)])[0]} for image_path in tqdm(image_paths)]

text_query = "man with red shirt"
vector = clip_model.get_text_embeddings([text_query])[0]

print("Searching for images...")
print("Query:", text_query)
idxs, scores = search_vector(vector, image_index, max_results=5)
for idx in idxs:
    img = load_image(image_paths[idx])
    show_image(img)
