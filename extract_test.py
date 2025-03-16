import os
import torch
import pickle
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from multiprocessing import Pool
from torch.cuda.amp import autocast

# Limit CPU threads for each process to prevent oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def process_scene_chunk(args):
    scene_chunk, input_folder, output_folder_gv, output_folder_dino, gpu_id, batch_size = args

    # Limit PyTorch CPU threads for this process
    torch.set_num_threads(1)
    
    # Set GPU device for this process
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)

    # Load processors and models for both feature extractors
    # Google Vit Model
    processor_gv = AutoImageProcessor.from_pretrained("google/vit-huge-patch14-224-in21k")
    model_gv = AutoModel.from_pretrained("google/vit-huge-patch14-224-in21k")
    model_gv.to(device)
    model_gv.eval()
    model_gv.half()  # use fp16

    # DinoV2 Model
    processor_dino = AutoImageProcessor.from_pretrained("timm/vit_large_patch14_dinov2.lvd142m")
    model_dino = AutoModel.from_pretrained("timm/vit_large_patch14_dinov2.lvd142m")
    model_dino.to(device)
    model_dino.eval()
    model_dino.half()  # use fp16

    # Process each scene in this chunk
    for scene in tqdm(scene_chunk, desc=f"GPU {gpu_id} Scenes", leave=True):
        scene_path = os.path.join(input_folder, scene)
        if not os.path.isdir(scene_path):
            continue

        features_gv = []    # for google vit features
        features_dino = []  # for dino features

        # Get sorted image files (assuming names like 0.jpg, 1.jpg, etc.)
        image_files = sorted(
            [f for f in os.listdir(scene_path) if f.endswith('.jpg')],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        batch_images = []
        for img_file in tqdm(image_files, desc=f"Scene {scene} images", leave=False):
            img_path = os.path.join(scene_path, img_file)
            image = Image.open(img_path).convert("RGB")
            batch_images.append(image)

            if len(batch_images) == batch_size:
                # Process batch for Google ViT
                inputs_gv = processor_gv(images=batch_images, return_tensors="pt")
                inputs_gv = {k: v.to(device) for k, v in inputs_gv.items()}

                # Process batch for DinoV2
                inputs_dino = processor_dino(images=batch_images, return_tensors="pt")
                inputs_dino = {k: v.to(device) for k, v in inputs_dino.items()}

                with torch.no_grad(), autocast():
                    outputs_gv = model_gv(**inputs_gv)
                    outputs_dino = model_dino(**inputs_dino)
                    # Average token features across the sequence dimension
                    batch_features_gv = outputs_gv.last_hidden_state.mean(dim=1)
                    batch_features_dino = outputs_dino.last_hidden_state.mean(dim=1)

                features_gv.extend(batch_features_gv.cpu())
                features_dino.extend(batch_features_dino.cpu())
                batch_images = []

        # Process any remaining images in the final batch
        if batch_images:
            inputs_gv = processor_gv(images=batch_images, return_tensors="pt")
            inputs_gv = {k: v.to(device) for k, v in inputs_gv.items()}
            inputs_dino = processor_dino(images=batch_images, return_tensors="pt")
            inputs_dino = {k: v.to(device) for k, v in inputs_dino.items()}
            with torch.no_grad(), autocast():
                outputs_gv = model_gv(**inputs_gv)
                outputs_dino = model_dino(**inputs_dino)
                batch_features_gv = outputs_gv.last_hidden_state.mean(dim=1)
                batch_features_dino = outputs_dino.last_hidden_state.mean(dim=1)
            features_gv.extend(batch_features_gv.cpu())
            features_dino.extend(batch_features_dino.cpu())

        # Concatenate features into a tensor [num_images, feature_size]
        scene_tensor_gv = torch.stack(features_gv, dim=0)
        scene_tensor_dino = torch.stack(features_dino, dim=0)

        # Save features to separate pickle files in their respective output directories
        save_path_gv = os.path.join(output_folder_gv, f'{scene}.pkl')
        with open(save_path_gv, 'wb') as f:
            pickle.dump(scene_tensor_gv, f)

        save_path_dino = os.path.join(output_folder_dino, f'{scene}.pkl')
        with open(save_path_dino, 'wb') as f:
            pickle.dump(scene_tensor_dino, f)

    return f"GPU {gpu_id} finished {len(scene_chunk)} scenes"

if __name__ == "__main__":
    # Define input folder and output folders for both models
    input_folder = "data/test_data/face_images"
    output_folder_gv = "data/test_data/googlevit"
    output_folder_dino = "data/test_data/dinov2large"
    os.makedirs(output_folder_gv, exist_ok=True)
    os.makedirs(output_folder_dino, exist_ok=True)

    # List all scene directories
    scenes = [scene for scene in os.listdir(input_folder)
              if os.path.isdir(os.path.join(input_folder, scene))]

    num_gpus = 4
    batch_size = 8  # Adjust based on your GPU memory and speed needs
    total_scenes = len(scenes)
    chunk_size = total_scenes // num_gpus
    chunks = []
    start = 0
    for i in range(num_gpus):
        end = start + chunk_size
        if i == num_gpus - 1:  # include any remaining scenes in the last chunk
            end = total_scenes
        chunks.append(scenes[start:end])
        start = end

    # Prepare arguments: each GPU gets its own contiguous chunk of scenes.
    args_list = [(chunk, input_folder, output_folder_gv, output_folder_dino, gpu_id, batch_size)
                 for gpu_id, chunk in enumerate(chunks)]

    # Process the chunks in parallel across 4 GPUs
    with Pool(processes=num_gpus) as pool:
        for result in pool.imap_unordered(process_scene_chunk, args_list):
            print(result)

