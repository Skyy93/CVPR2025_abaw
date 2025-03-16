import os
#os.environ['LD_LIBRARY_PATH'] = '/home/hallmeto/miniforge3/envs/abaw8/lib/python3.12/site-packages/nvidia/cudnn/lib/:' + os.environ['LD_LIBRARY_PATH']
from tqdm import tqdm
import whisperx
import pandas as pd
# Check CUDA availability for Whisper
device = "cuda"
model_size = "large-v3"
lang = 'en'
# Load the Whisper model on the specified device
model = whisperx.load_model(model_size, device, compute_type='float16', language=lang)
model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
        
# Ensure the output directory exists
output_dir = "data/text/"
os.makedirs(output_dir, exist_ok=True)

# Path to your audio files
#audio_dir = "data/audio/"
audio_dir = '/mnt/datasets/ABAW/8th/BAH/BAH-train-8th-ABAW/ABAW-8th-BAH-train-data/audios'

# List all .mp3 files in the audio directory
#audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]#[:3]
audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]#[:3]
df = None

def _fix_missing_timestamps(data):
            """
            https://github.com/m-bain/whisperX/issues/253
            Some characters might miss timestamps and recognition scores. This function adds estimated time stamps assuming a fixed time per character of 65ms.
            Confidence for each added timestamp will be 0.
            Args:
                data (dictionary): output dictionary as returned by process_data
            """
            last_end = 0
            for s in data["segments"]:
                for w in s["words"]:
                    if "end" in w.keys():
                        last_end = w["end"]
                    else:
                        #TODO: rethink lower bound for confidence; place word centred instead of left aligned
                        w["start"] = last_end
                        last_end += 0.065
                        w["end"] = last_end
                        #w["score"] = 0.000
                        #w['score'] = _hmean([x['score'] for x in s['words'] if len(x) == 4])

# Initialize tqdm progress bar
with tqdm(total=len(audio_files), desc="Transcribing", unit="file") as pbar:
    for filename in audio_files:
        # Construct the full path to the current file
        audio_path = os.path.join(audio_dir, filename)
        
        # Transcribe the audio file
        
        result = model.transcribe(audio_path, batch_size=32)
        result_a = whisperx.align(result["segments"], model_a, metadata, audio_path, device, return_char_alignments=False)
        
        _fix_missing_timestamps(result_a)
        
        start, end, word, cumword = [], [], [], []
        for res in result_a['word_segments']:
            start.append(res['start'])
            end.append(res['end'])
            word.append(res['word'])
            cumword.append(' '.join(word))

        if df is None:
            df = pd.DataFrame.from_dict({'file': os.path.splitext(audio_path)[0].split(os.sep)[-1], 'start': start, 'end': end, 'word': word, 'cumword': cumword})
        else:
            df = pd.concat([df, pd.DataFrame.from_dict({'file': os.path.splitext(audio_path)[0].split(os.sep)[-1], 'start': start, 'end': end, 'word': word, 'cumword': cumword})], axis=0)

        # Extract the base filename without the extension and add .txt
        #text_filename = os.path.splitext(filename)[0] + ".txt"
        
        # Construct the full path for the output text file
        #text_path = os.path.join(output_dir, text_filename)
        # Save the transcription to a text file
        #with open(text_path, "w") as text_file:
        #    text_file.write(result)
        
        pbar.update(1)  # Update progress bar
    df.reset_index().to_pickle('ah-whisperx.xz')

print("All files have been transcribed.")  # 82665_Question_1_2024-11-13_17-23-28_Video is empty
