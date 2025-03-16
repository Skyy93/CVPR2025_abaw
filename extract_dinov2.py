import os
import torch
import pickle
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from multiprocessing import Pool
from torch.cuda.amp import autocast
import cv2


os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
cv2.setNumThreads(2)

def process_scene_chunk(args):
    torch.set_num_threads(2)
    scene_chunk, input_folder, output_folder, gpu_id, batch_size = args

    # Set device for this process
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)

    # Load Hugging Face processor and model for dinov2
    processor = AutoImageProcessor.from_pretrained("timm/vit_large_patch14_dinov2.lvd142m")
    model = AutoModel.from_pretrained("timm/vit_large_patch14_dinov2.lvd142m")
    model.to(device)
    model.eval()

    # Optionally enable fp16 mode on the model by converting it to half precision
    model.half()

    for scene in tqdm(scene_chunk, desc=f"GPU {gpu_id} Scenes", leave=True):
        scene_path = os.path.join(input_folder, scene)
        if not os.path.isdir(scene_path):
            continue

        features = []
        # Get sorted image files
        image_files = sorted(
            [f for f in os.listdir(scene_path) if f.endswith('.jpg')],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        # Process images in batches
        batch_images = []
        batch_file_names = []  # to maintain order if needed

        for img_file in tqdm(image_files, desc=f"Scene {scene} images", leave=False):
            img_path = os.path.join(scene_path, img_file)
            image = Image.open(img_path).convert("RGB")
            batch_images.append(image)
            batch_file_names.append(img_file)

            if len(batch_images) == batch_size:
                # Preprocess the batch of images
                inputs = processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad(), autocast():
                    outputs = model(**inputs)
                    # Average the token features for each image in the batch
                    batch_features = outputs.last_hidden_state.mean(dim=1)
                # Append features for each image in the batch (move to CPU)
                for feat in batch_features:
                    features.append(feat.cpu())
                batch_images = []
                batch_file_names = []

        # Process any remaining images in the batch
        if batch_images:
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad(), autocast():
                outputs = model(**inputs)
                batch_features = outputs.last_hidden_state.mean(dim=1)
            for feat in batch_features:
                features.append(feat.cpu())

        # Concatenate features for the scene: [num_images, feature_size]
        scene_tensor = torch.stack(features, dim=0)
        # Save tensor as a pickle file
        save_path = os.path.join(output_folder, f'{scene}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(scene_tensor, f)

    return f"GPU {gpu_id} finished {len(scene_chunk)} scenes"

if __name__ == "__main__":
    # Define paths
    input_folder = "data/face_images"
    output_folder = "data/dinov2large"
    os.makedirs(output_folder, exist_ok=True)

    # List all scene directories
    scenes = [scene for scene in os.listdir(input_folder)
              if os.path.isdir(os.path.join(input_folder, scene))]

    num_gpus = 4
    batch_size = 8  # Adjust as needed
    total_scenes = len(scenes)
    chunk_size = total_scenes // num_gpus
    chunks = []
    start = 0
    for i in range(num_gpus):
        end = start + chunk_size
        if i == num_gpus - 1:  # include remainder in the last chunk
            end = total_scenes
        chunks.append(scenes[start:end])
        start = end

    # Prepare argument list: each GPU gets its contiguous chunk of scenes.
    args_list = [(chunk, input_folder, output_folder, gpu_id, batch_size) 
                 for gpu_id, chunk in enumerate(chunks)]

    # Use a multiprocessing pool with one process per GPU
    with Pool(processes=num_gpus) as pool:
        for result in pool.imap_unordered(process_scene_chunk, args_list):
            print(result)
 
