import os
import torch
import openl3
import pandas as pd
import soundfile as sf


root_dir = './UrbanSound8K/audio/'
metadata_file = './UrbanSound8K/metadata/UrbanSound8K.csv'
df = pd.read_csv(metadata_file)

n_files = 200

for f in sorted(df['fold'].unique()):
    print(f'Fold: {f}')
    files = []
    labels = []
    srs = []
    df_inner = df[df['fold'] == f].reset_index(drop=True)
    for idx, row in df_inner.iterrows():
        file_name = os.path.join(os.path.abspath(root_dir), 'fold' + str(row['fold']) + '/', row['slice_file_name'])
        audio, sr = sf.read(file_name)
        files.append(audio)
        labels.append(row['classID'])
        srs.append(sr)
        if ((idx % n_files == 0) and (idx != 0)) or (idx + 1 == len(df_inner)):
            emb_list, ts_list = openl3.get_audio_embedding(files, srs, content_type="env", batch_size=256)
            encode_dataset = torch.tensor([])
            for e in emb_list:
                encoding = torch.tensor(e.mean(axis=0))
                encode_dataset = torch.cat((encode_dataset, encoding))
            torch.save(encode_dataset, f'emb_{f}_{idx}.pt')
            torch.save(torch.tensor(labels), f'lbl_{f}_{idx}.pt')
            files = []
            srs = []
