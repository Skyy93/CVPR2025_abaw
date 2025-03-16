import yaml
import pandas as pd

print('loading yaml...')
with open('/mnt/datasets/ABAW/8th/BAH/BAH-train-8th-ABAW/ABAW-8th-BAH-train-data/annotation.yml', 'r') as f:
    y = yaml.safe_load(f)

# key = path, e.g. '82557/Visite 1/82557_Question_1_2024-08-22 14-46-11_Video.mp4'
# subkeys
# dict_keys(['fr_detailed_ah', 'frame_annotation', 'global_ah', 'time_detailed_ah'])

print('converting...')
df = None
for k, v in y.items():
    frame = [int(x[0].split('frame-')[-1][:-4]) for x in v['frame_annotation']]
    ah = [x[1] for x in v['frame_annotation']]
    if df is None:
        df = pd.DataFrame.from_dict({'file': k.split('/')[-1].split('.')[0], 'frame': frame, 'ah': ah})
    else:
        df = pd.concat([df, pd.DataFrame.from_dict({'file': k.split('/')[-1].split('.')[0], 'frame': frame, 'ah': ah})], axis=0)

df.reset_index().to_pickle('ah-frame-annotation.xz')
print('done')