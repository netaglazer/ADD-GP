import torch
import random
import librosa
import math
import numpy as np
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from os import path
import os

def pad_sequence(batch):
    """Pad a batch of variable-length sequences to the length of the longest sequence."""
    max_size = max([s.size(0) for s, _ in batch])
    batch_padded = [torch.nn.functional.pad(s, (0, max_size - s.size(0))) for s, _ in batch]
    sequences_padded = torch.stack(batch_padded)
    labels = torch.tensor([label for _, label in batch])
    return sequences_padded, labels


def sample_audio(audio, target_length=32000):
    """
    Samples a subarray of specified length from the audio array.

    Args:
        audio (numpy.ndarray): The audio array.
        target_length (int): The desired length of the subarray.

    Returns:
        numpy.ndarray: The sampled subarray.
    """
    if len(audio) <= target_length:
        return audio  # If audio is already shorter, return as is.

    # Randomly select a starting index
    start_idx = np.random.randint(0, len(audio) - target_length + 1)
    return audio[start_idx:start_idx + target_length]

def custom_collate(batch):
    """Collate function that groups audio file paths and labels without padding."""
    representation = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return representation, labels

def normalize_audio(audio):
    """Normalize audio to have zero mean and unit variance."""
    return (audio - audio.mean()) / audio.std()

def is_nan_or_string(value):
    # Check if value is NaN
    if isinstance(value, float) and math.isnan(value):
        return True
    # Check if value is a string
    if isinstance(value, str):
        return True
    # Otherwise, return False
    return False

class DeepFakeDetectionDataset(Dataset):
    def __init__(self, data_path, tts_models, few_shot_samples=None, eleven_labs=False, type='train', id=None):
        self.data_path = data_path
        self.id = id
        self.eleven_labs = eleven_labs
        if self.eleven_labs:
            self.eleven_labs_path = 'data/Libri_dataset_11labs_new_train.csv'
            self.eleven_labs_df = pd.read_csv(self.eleven_labs_path)
            self.eleven_labs_df_1 = self.eleven_labs_df[self.eleven_labs_df['label'] ==1].sample(few_shot_samples, random_state=54)
            self.eleven_labs_df_0 = self.eleven_labs_df[self.eleven_labs_df['label'] ==0].sample(few_shot_samples, random_state=36)
            self.eleven_labs_df = pd.concat([self.eleven_labs_df_1, self.eleven_labs_df_0]).reset_index(drop=True)
            print(self.eleven_labs_df['label'].value_counts())
        self.df = pd.DataFrame({})
        self.load_data(data_path, type)
        self.tts_models = tts_models
        self.type = type

    def load_data(self, data_path, type):
        data = pd.read_csv(data_path)
        if type == 'train':
            self.df = data
            print('data_path', data_path)
            data = pd.read_csv(data_path)
            print('shape(self.df)', self.df.shape)
        elif type == 'val':
            self.df = data.sample(1000).reset_index(drop=True)
            self.df['is_11labs'] = self.df['eleven_labs_clone'].apply(lambda x: os.path.isfile(x))
            self.df = self.df[self.df['is_11labs'] == True]
            self.df_1 = self.df[self.df['label'] == 1]
            self.df_0 = self.df[self.df['label'] == 0]
            self.df = pd.concat([self.df_0,self.df_1.sample(int(0.75*len(self.df_1)))])


        if type == 'eleven_labs':
            self.df = pd.read_csv(data_path)
            self.df = self.df.sample(100)
            self.df = self.df.reset_index(drop=True)
            self.df['11labs'] = self.df['f5_clone'].apply(lambda x: x.replace('_F5', 'eleven_labs'))
            self.df['is_11labs'] = self.df['11labs'].apply(lambda x: path.isfile(x))
            self.df = self.df[self.df['is_11labs']==False].reset_index(drop=True)
            self.df = pd.concat([self.eleven_labs_df_0, self.eleven_labs_df_0,
                                 self.eleven_labs_df,  self.eleven_labs_df_0,
                                 self.eleven_labs_df,  self.eleven_labs_df_0,
                                 self.eleven_labs_df_0, self.eleven_labs_df_0, self.eleven_labs_df, self.df]).reset_index(drop=True)
        self.df = self.df.reset_index(drop=True)


    def __getitem__(self, idx):
        label = self.df.loc[idx]['label']

        if label == 1:
            wav_sample_path = self.df.loc[idx]['path']
            audio, sample_rate = librosa.load(wav_sample_path)
        else:
            if  self.eleven_labs and isinstance(self.df.loc[idx][f'eleven_labs_clone'], str) and path.isfile(self.df.loc[idx][f'eleven_labs_clone']):
                # print('eleven_labs')
                wav_sample_path = self.df.loc[idx][f'eleven_labs_clone']
                audio, sample_rate = librosa.load(wav_sample_path)

                target_sample_rate = 16000
                if sample_rate != target_sample_rate:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
                    sample_rate = target_sample_rate

                audio = normalize_audio(audio)

                rand = random.random()
                if rand > 0.8:
                    audio = sample_audio(audio, target_length=56000)
                elif rand > 0.3:
                    tts_model = random.choice(self.tts_models)
                    try:
                        new_wav_sample_path = self.df.loc[idx][f'{tts_model}_clone']
                        new_audio, new_sample_rate = librosa.load(new_wav_sample_path)
                    except:
                        new_wav_sample_path = self.df.loc[idx][f'yourtts_clone']
                        new_audio, new_sample_rate = librosa.load(new_wav_sample_path)
                    target_sample_rate = 16000
                    if sample_rate != target_sample_rate:
                        new_audio = librosa.resample(new_audio, orig_sr=new_sample_rate, target_sr=target_sample_rate)
                        new_sample_rate = target_sample_rate

                    audio = normalize_audio(audio)[:56000]
                    new_audio = normalize_audio(new_audio)[:56000]
                    min_len = min(len(audio), len(new_audio))

                    new_rand = random.random()
                    # if new_rand > 0.6:
                    #     audio = 0.5*(new_audio[:min_len] + audio[:min_len])
                    weight = random.random()  # Random number between 0 and 1
                    audio = (weight * new_audio[:min_len] + (1 - weight) * audio[:min_len])
                    # elif new_rand > 0.3:
                    #     audio = ((3/4)*new_audio[:min_len] + (1/4)*audio[:min_len])
                    # else:
                    #     audio = ((1/4)*new_audio[:min_len] + (3/4)*audio[:min_len])
                else:  # rand > 0.2:
                    audio = audio[:56000]
                return audio, label
            else:
                tts_model = random.choice(self.tts_models)
                try:
                    wav_sample_path = self.df.loc[idx][f'{tts_model}_clone']
                    audio, sample_rate = librosa.load(wav_sample_path)
                except:
                    print(wav_sample_path, 'does not exist')
                    label = 1
                    wav_sample_path = self.df.loc[idx]['path']
                    # wav_sample_path = self.df.loc[idx][f'yourtts_clone']
                    audio, sample_rate = librosa.load(wav_sample_path)


        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate

        audio = normalize_audio(audio)
        audio = audio[:56000]

        directory = "/data/openai/OpenAI"
        file_list = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_list.append(os.path.join(root, file))

        return audio, label

    def read_audio(self, audio_path):
        signal, sr = torchaudio.load(audio_path)
        return signal, sr

    def __len__(self):
        return len(self.df)

    def extract_embedding(self, audio_path):

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        # Convert waveform to the required format
        audio = waveform.squeeze().numpy().astype(np.float32)

        with torch.no_grad():
            self.processor.tokenizer.set_prefix_tokens(language='en')
            input_features = self.processor(
                torch.tensor(audio, dtype=torch.float32),
                return_tensors="pt",
                sampling_rate=16000,  # dtype=torch.float16
            ).input_features

            decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id
            last_hidden_state = self.model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state            # Process the audio with Whisper to extract embeddings
            embedding = last_hidden_state.mean(dim=1)  # Aggregate embedding information

        return embedding, input_features

