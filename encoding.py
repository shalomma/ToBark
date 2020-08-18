import os
import openl3
import pandas as pd
import soundfile as sf

root_dir = './UrbanSound8K/audio/'
metadata_file = './UrbanSound8K/metadata/UrbanSound8K.csv'
df = pd.read_csv(metadata_file)

files = []
srs = []
for idx, row in df.iterrows():
    file_name = os.path.join(os.path.abspath(root_dir), 'fold' + str(row['fold']) + '/', row['slice_file_name'])
    audio, sr = sf.read(file_name)
    files.append(audio)
    srs.append(sr)

emb_list, ts_list = openl3.get_audio_embedding(files, srs, batch_size=256)
