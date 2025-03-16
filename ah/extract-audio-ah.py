from pathlib import Path
from subprocess import Popen
from tqdm.contrib.concurrent import process_map
import os


def func(path: Path):
    #os.environ['TMP'] = str(path.parent)
    po1 = Popen(f'ffmpeg -y -hide_banner -loglevel error -i "{path}" -ac 1 -ar 16000 "{path.parents[3] / 'audios' / (path.stem + '.wav')}"', shell=True)
    #po2 = Popen(f'ffmpeg-normalize {path} -o {path.parent}/therapist.audio.wav -f -ac 1 -ar 48000', shell=True)
    po1.wait()
    #po2.wait()        
    
if __name__ == '__main__':
    p = Path('/mnt/datasets/ABAW/8th/BAH/BAH-train-8th-ABAW/ABAW-8th-BAH-train-data')
    (p / 'audios').mkdir(exist_ok=True)
    videos = sorted(list(p.glob('videos/*/*/*.mp4')))
    #print(folders, os.environ['TMP'])
    #func(videos[0])
    process_map(func, videos, total=len(videos))