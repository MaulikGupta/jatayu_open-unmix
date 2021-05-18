import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
from tqdm import tqdm
import time

def check_dir(save_dir):
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

umx_path = '/datasets'

crow_call = os.listdir(os.path.join(umx_path, 'crow', 'mp3'))
check_dir(os.path.join(umx_path, 'crow', 'wav'))

# convert mp3 --> wav

for call in crow_call:
    name = call.split('.')[0]
    mp3_path = os.path.join(umx_path, 'crow', 'mp3', call)
    wav_path = os.path.join(umx_path, 'crow', 'wav', name+'.wav')

    if os.path.exists(wav_path) is False:
        os.system('ffmpeg -i {} -vn -acodec pcm_s16le -ac 1 -ar 44100 -f wav {}'\
            .format(mp3_path, wav_path))

# train test split for crows

crow_call = sorted(os.listdir(os.path.join(umx_path, 'crow', 'wav')))
train_path = os.path.join(umx_path, 'data', 'train', 'crow_call')
valid_path = os.path.join(umx_path, 'data', 'valid', 'crow_call')

train_fns, valid_fns = train_test_split(crow_call, test_size=0.2, random_state=0)
print(train_fns)
print(valid_fns)

for split, fns in zip(['train', 'valid'], [train_fns, valid_fns]):
    interval = 44100 * 5
    for fn in tqdm(fns):
        wav_path = os.path.join(umx_path, 'crow', 'wav', fn)
        rate, wav = wavfile.read(wav_path)
        # wav = wav[rate*600]
        stop = (wav.shape[0]//interval)*interval
        for i in range(0, stop-interval, interval):
            sample = wav[i:i+interval]
            save_dir = os.path.join(umx_path, 'data', split, 'crow_call')
            check_dir(save_dir)
            fn = fn.split('.')[0]
            i = int(i / interval)
            save_fn = str(os.path.join(save_dir, fn+'_{}.wav'.format(i)))
            if os.path.exists(save_fn) is True:
                continue
            wavfile.write(filename = save_fn,rate = rate,data = sample)

# train test split for sparrows

sparrow_call = sorted(os.listdir(os.path.join(umx_path, 'sparrow', 'wav')))
train_path = os.path.join(umx_path, 'data', 'train', 'sparrow_call')
valid_path = os.path.join(umx_path, 'data', 'valid', 'sparrow_call')

train_fns, valid_fns = train_test_split(sparrow_call, test_size=0.2, random_state=0)
print(train_fns)
print(valid_fns)

for split, fns in zip(['train', 'valid'], [train_fns, valid_fns]):
    interval = 44100 * 5
    for fn in tqdm(fns):
        wav_path = os.path.join(umx_path, 'sparrow', 'wav', fn)
        rate, wav = wavfile.read(wav_path)
        # wav = wav[rate*600]
        stop = (wav.shape[0]//interval)*interval
        for i in range(0, stop-interval, interval):
            sample = wav[i:i+interval]
            save_dir = os.path.join(umx_path, 'data', split, 'sparrow_call')
            check_dir(save_dir)
            fn = fn.split('.')[0]
            i = int(i / interval)
            save_fn = str(os.path.join(save_dir, fn+'_{}.wav'.format(i)))
            if os.path.exists(save_fn) is True:
                continue
            wavfile.write(filename = save_fn,rate = rate,data = sample)

# train test split for titlis

titli_call = sorted(os.listdir(os.path.join(umx_path, 'titli', 'wav')))
train_path = os.path.join(umx_path, 'data', 'train', 'titli_call')
valid_path = os.path.join(umx_path, 'data', 'valid', 'titli_call')

train_fns, valid_fns = train_test_split(titli_call, test_size=0.2, random_state=0)
print(train_fns)
print(valid_fns)

for split, fns in zip(['train', 'valid'], [train_fns, valid_fns]):
    interval = 44100 * 5
    for fn in tqdm(fns):
        wav_path = os.path.join(umx_path, 'titli', 'wav', fn)
        rate, wav = wavfile.read(wav_path)
        # wav = wav[rate*600]
        stop = (wav.shape[0]//interval)*interval
        for i in range(0, stop-interval, interval):
            sample = wav[i:i+interval]
            save_dir = os.path.join(umx_path, 'data', split, 'titli_call')
            check_dir(save_dir)
            fn = fn.split('.')[0]
            i = int(i / interval)
            save_fn = str(os.path.join(save_dir, fn+'_{}.wav'.format(i)))
            if os.path.exists(save_fn) is True:
                continue
            wavfile.write(filename = save_fn,rate = rate,data = sample)

# train, test, splits for interferers
df = pd.read_csv(os.path.join(umx_path, 'ESC-50', 'meta', 'esc50.csv'))
print(df.head(10))

inter_dirs = np.unique(df.category).tolist()

for cat in inter_dirs:
    print('copying files from {} directory'.format(cat))
    df_cat = df[df.category==cat]
    train_df, valid_df = train_test_split(df_cat, test_size=0.1, random_state=0)
    for split, split_df in zip(['train', 'valid'], [train_df, valid_df]):
        dir_path = os.path.join(umx_path, 'data', split, 'interfer')
        check_dir(dir_path)

        # iterate splits and save wav file
        for i, row in split_df.iterrows():
            fn = row['filename']
            fn_path = os.path.join(dir_path, fn)
            if os.path.exists(fn_path) is True:
                continue
            src_path = os.path.join(umx_path, 'ESC-50', 'audio', fn)
            os.system('cp {} {}'.format(src_path, fn_path))
