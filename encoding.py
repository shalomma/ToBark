import os
import torch
import openl3
import pandas as pd
import numpy as np
import soundfile as sf
import scipy.stats as stats


root_dir = './UrbanSound8K/audio/'
metadata_file = './UrbanSound8K/metadata/UrbanSound8K.csv'
df = pd.read_csv(metadata_file)

n_files = 300

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
            emb_list, ts_list = openl3.get_audio_embedding(files, srs, embedding_size=512,
                                                           content_type="env", batch_size=256)
            encode_dataset = torch.tensor([])
            for i, e in enumerate(emb_list):
                mean_ = torch.tensor(e.mean(axis=0))
                median_ = torch.tensor(np.median(e, axis=0))
                max_ = torch.tensor(e.max(axis=0))
                min_ = torch.tensor(e.min(axis=0))
                skew_ = torch.tensor(stats.skew(e, axis=0))
                kurtosis_ = torch.tensor(stats.kurtosis(e, axis=0))
                entropy_ = torch.tensor(stats.entropy(e, axis=0))
                encoding = torch.stack([mean_, median_, max_, min_, skew_, kurtosis_, entropy_]).view(1, 7, 512)
                if i == 0:
                    encode_dataset = encoding
                else:
                    encode_dataset = torch.cat((encode_dataset, encoding))

            torch.save(encode_dataset, f'emb_{f}_{idx}.pt')
            torch.save(torch.tensor(labels), f'lbl_{f}_{idx}.pt')
            files = []
            labels = []
            srs = []
