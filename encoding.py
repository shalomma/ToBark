import os
import openl3
import pandas as pd
import soundfile as sf

root_dir = './UrbanSound8K/audio/'
metadata_file = './UrbanSound8K/metadata/UrbanSound8K.csv'
df = pd.read_csv(metadata_file)

for idx, row in df.iterrows():
    file_name = os.path.join(os.path.abspath(root_dir), 'fold' + str(row['fold']) + '/', row['slice_file_name'])
    audio, sr = sf.read(file_name)
    emb, ts = openl3.get_audio_embedding(audio, sr)
    row['sr'] = sr
    row['ts'] = ts
    row['raw_len'] = audio.shape[0]
    row['ch'] = audio.shape[1]
    row['emb_len'] = emb.shape[1]

df.to_csv('temp.csv')
