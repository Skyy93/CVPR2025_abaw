import os
from tqdm import tqdm
from faster_whisper import WhisperModel
# Check CUDA availability for Whisper
device = "cuda"
model_size = "large-v3"
# Load the Whisper model on the specified device
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Ensure the output directory exists
output_dir = "data/text/"
os.makedirs(output_dir, exist_ok=True)

# Path to your audio files
audio_dir = "data/audio/"

# List all .mp3 files in the audio directory
audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]

# Initialize tqdm progress bar
with tqdm(total=len(audio_files), desc="Transcribing", unit="file") as pbar:
    for filename in audio_files:
        # Construct the full path to the current file
        audio_path = os.path.join(audio_dir, filename)
        
        # Transcribe the audio file
        result,_ = model.transcribe(audio_path, beam_size=5)
        try:
            result = list(result)[0].text
        except:
            result = ""
        # Extract the base filename without the extension and add .txt
        text_filename = os.path.splitext(filename)[0] + ".txt"
        
        # Construct the full path for the output text file
        text_path = os.path.join(output_dir, text_filename)
        # Save the transcription to a text file
        with open(text_path, "w") as text_file:
            text_file.write(result)
        
        pbar.update(1)  # Update progress bar

print("All files have been transcribed.")
