import os
import torch
import pickle
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

# Paths
input_folder = 'data/face_images'
output_folder = 'data/googlevit'

# Create output directory if not exists
os.makedirs(output_folder, exist_ok=True)

# Load DinoV2 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = AutoImageProcessor.from_pretrained('google/vit-huge-patch14-224-in21k')
model = AutoModel.from_pretrained('google/vit-huge-patch14-224-in21k').to(device)
model.eval()

# Process each scene
for scene in tqdm(os.listdir(input_folder), desc='Scenes'):
    scene_path = os.path.join(input_folder, scene)
    if not os.path.isdir(scene_path):
        continue

    features = []

    # Sort files numerically (0.jpg, 1.jpg, etc.)
    image_files = sorted([f for f in os.listdir(scene_path) if f.endswith('.jpg')],
                         key=lambda x: int(os.path.splitext(x)[0]))

    for img_file in tqdm(image_files, desc=f'Processing scene {scene}', leave=False):
        img_path = os.path.join(scene_path, img_file)
        image = Image.open(img_path).convert('RGB')

        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            feature_vector = outputs.last_hidden_state.mean(dim=1)  # [1, feature_size]

        features.append(feature_vector.cpu())

    # Stack features [sequence_length, feature_size]
    scene_tensor = torch.cat(features, dim=0)

    # Save tensor as pickle file
    save_path = os.path.join(output_folder, f'{scene}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(scene_tensor, f)

print("Encoding completed successfully.")

