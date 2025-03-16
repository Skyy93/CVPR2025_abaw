import os
import pandas as pd

audio_files = os.listdir('data/test_data/raw/')
filenames = [os.path.splitext(file)[0] for file in audio_files if file.endswith('.mp4')]

df = pd.DataFrame(filenames, columns=['Filename'])

for column in ['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']:
    df[column] = 0.0

csv_path = 'data/test_data/test_split.csv'
print(len(filenames))
df.to_csv(csv_path, index=False)

