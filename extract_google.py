import os
import re
import torch
import pickle
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

# Paths
input_folder = '/Coding/CVPR2025_abaw_framewise/data_abaw/test_data/cropped-aligned-faces'
output_folder = '/Coding/CVPR2025_abaw_framewise/data_abaw/test_data/googlevit'

# Create output directory if not exists
os.makedirs(output_folder, exist_ok=True)

# Load DinoV2 model
device = 'cuda:14' if torch.cuda.is_available() else 'cpu'
processor = AutoImageProcessor.from_pretrained('google/vit-huge-patch14-224-in21k')
model = AutoModel.from_pretrained('google/vit-huge-patch14-224-in21k').to(device)
model.eval()

def get_frame_number(filename):
    match = re.search(r'frame-(\d+)', filename)
    return int(match.group(1)) if match else -1

# Recursively walk through all subdirectories in the input folder
for root, dirs, files in os.walk(input_folder):
    # Look for files that start with "frame-" and end with ".jpg"
    image_files = [f for f in files if f.endswith('.jpg') and f.startswith('frame-')]
    if image_files:
        # Sort the images based on the frame number
        image_files = sorted(image_files, key=get_frame_number)
        features = []

        # Get the relative path from the input folder
        rel_path = os.path.relpath(root, input_folder)

        # Create the corresponding output directory
        output_video_folder = os.path.join(output_folder, rel_path)
        os.makedirs(output_video_folder, exist_ok=True)

        # Create the pickle file name path based on the video folder name
        video_folder_name = os.path.basename(root)
        pickle_filename = f"{video_folder_name.removesuffix(".mp4")}.pkl"
        save_path = os.path.join(output_video_folder, pickle_filename)

        # If the pickle file already exists, skip processing this folder
        if os.path.exists(save_path):
            print(f"Features already exist at {save_path}. Skipping...")
            continue

        # Process each image in the current folder (video folder)
        for img_file in tqdm(image_files, desc=f'Processing folder {root}', leave=False):
            img_path = os.path.join(root, img_file)
            image = Image.open(img_path).convert('RGB')

            # Prepare the image using the processor
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # Use the mean of the last hidden state as the feature vector
                feature_vector = outputs.last_hidden_state.mean(dim=1)  # [1, feature_size]

            features.append(feature_vector.cpu())

        # Stack the features into a tensor of shape [sequence_length, feature_size]
        scene_tensor = torch.cat(features, dim=0)

        # Save the tensor as a pickle file
        with open(save_path, 'wb') as f:
            pickle.dump(scene_tensor, f)

        print(f"Saved features to {save_path}")

print("Encoding completed successfully.")