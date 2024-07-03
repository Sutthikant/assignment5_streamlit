import os
import torch
import faiss
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_index.bin'))
index = faiss.read_index(index_path)

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train_cat_breeds.csv'))
df_train = pd.read_csv(data_path)
image_paths = []
for img_id, breed in zip(df_train['id'].values, df_train['breed'].values):
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'images', breed, f'{str(img_id)}.jpg'))
    image_paths.append(img_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def query(text_input):
    text_tokens = processor(text=text_input, return_tensors="pt", padding=True).input_ids.to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(text_tokens).cpu().numpy()

    _, indices = index.search(text_features, 1)
    img_idx = indices[0][0]

    img_path = image_paths[img_idx]
    img = Image.open(img_path)
    
    return img
