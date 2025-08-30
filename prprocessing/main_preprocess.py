import os
import torchaudio
import torch
from torchaudio.transforms import Resample
import soundfile as sf
import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale


def segment_audio(record_path,duration=1):
    try:
        waveform,sample_rate=torchaudio.load(record_path)
        resempler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resempler(waveform)
        sample_rate = 16000
        total_segments = waveform.size(1) // (sample_rate * duration)
        if total_segments == 0:
            print(f"Audio file {record_path} is too short for segmentation.")
            return []
        segments = []
        for i in range(total_segments):
            start = i * sample_rate * duration
            end = start + sample_rate * duration
            if end > waveform.size(1):
                segment = waveform[:, start:waveform.size(1)]
                segment = torch.nn.functional.pad(segment, (0, end - waveform.size(1)))
            segment = waveform[:, start:end]
            segments.append(segment)
        return segments
    except Exception as e:
        print(f"Error processing {record_path}: {e}")
        return []
def get_mfcc(segment, sample_rate=16000, n_mfcc=13):
    # accept numpy or torch input and ensure dtype=float32 and shape [channel, time]
    if isinstance(segment, np.ndarray):
        waveform = torch.from_numpy(segment.astype(np.float32))
    elif isinstance(segment, torch.Tensor):
        waveform = segment.to(dtype=torch.float32)
    else:
        waveform = torch.tensor(np.asarray(segment, dtype=np.float32))
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 23, "center": False}
    )
    mfcc = mfcc_transform(waveform)
    mfcc_mean = mfcc.mean(dim=2).squeeze().numpy()
    return mfcc_mean
def get_features(segmnt,f0min=75, f0max=600, unit="Hertz"):
    pitch = call(segmnt, "To Pitch", 0.0, f0min, f0max)
    pitch_values = pitch.selected_array['frequency']
    putch_values= pitch_values[pitch_values > 0]
    perceived_pitch = np.median(putch_values) if len(putch_values) > 0 else 0
    meanF0 = call(pitch, "Get mean", 0, 0,unit)
    harmonicity = call(segmnt, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess=call(segmnt, "To PointProcess (periodic, cc)", f0min, f0max)

    localjitter = call(pointProcess, "Get jitter (local)", 0.0, 0.0,0.0001, 0.02,1.3)
    localAbsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0.0, 0.0,0.0001, 0.02,1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0.0, 0.0,0.0001, 0.02,1.3)
    ddqJitter = call(pointProcess, "Get jitter (ddp)", 0.0, 0.0,0.0001, 0.02,1.3)

    localShimer= call([segmnt,pointProcess], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    localDbShimmer = call([segmnt,pointProcess], "Get shimmer (local_dB)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([segmnt,pointProcess], "Get shimmer (apq3)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    apq5Shimmer = call([segmnt,pointProcess], "Get shimmer (apq5)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    f="""
    formants= call(segmnt, "To Formant (burg)", 0.0, 5, 5000, 0.025, 50)
    numPoints=call(pointProcess, "Get number of points")
    f1_list, f2_list, f3_list, f4_list = [], [], [], []
    for point in range(1, numPoints + 1):
        time = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, time, unit, 'Linear')
        f2 = call(formants, "Get value at time", 2, time, unit, 'Linear')
        f3 = call(formants, "Get value at time", 3, time, unit, 'Linear')
        f4 = call(formants, "Get value at time", 4, time, unit, 'Linear')
        if f1 > 0: f1_list.append(f1)
        if f2 > 0: f2_list.append(f2)
        if f3 > 0: f3_list.append(f3)
        if f4 > 0: f4_list.append(f4)
    f=np.median(f1_list+f2_list+f3_list+f4_list)/4
    """
    # build a float32 torch tensor from parselmouth.Sound values for MFCC
    samples = np.asarray(segmnt.values, dtype=np.float32)
    mfcc_mean = get_mfcc(torch.from_numpy(samples).unsqueeze(0))


    features={
        "localAbsoluteJitter": localAbsoluteJitter,
        "localjitter": localjitter,
        "rapJitter": rapJitter,
        "ddqJitter": ddqJitter,
        "localShimer": localShimer,
        "localDbShimmer": localDbShimmer,
        "apq3Shimmer": apq3Shimmer,
        "apq5Shimmer": apq5Shimmer,
        "hnr": hnr,
        "FundamentalFrequency": perceived_pitch,
        "meanF0": meanF0}
    for i in range(13):
        features[f'mfcc_{i+1}'] = mfcc_mean[i]

    return features

def process_audio_file(record_path,segment_duration=1):
    segments = segment_audio(record_path, duration=segment_duration)
    record_features_df=pd.DataFrame()
    for i, segment in enumerate(segments):
        sf.write('temp_segment.wav', segment.numpy().T, 16000)
        segmnt = parselmouth.Sound('temp_segment.wav')
        features = get_features(segmnt)
        features['path']=record_path
        features['audio_id'] = record_path.split('/')[-1].split('.')[0][3:]
        features['segment'] = i + 1
        record_features_df = pd.concat([record_features_df, pd.DataFrame([features])], ignore_index=True)
        os.remove('temp_segment.wav')

    return record_features_df
def scale_features(df, feature_columns):
   
    for col in feature_columns:
        if col in df.columns:
            df[col] = minmax_scale(df[col])
    return df

def preprocess_dataset(data_dir, output_csv, segment_duration=1):
    all_features_df = pd.DataFrame()
    i=0
    for subfolder in os.listdir(data_dir):
        if subfolder.endswith('.xlsx'):
            continue
        subfolder_features_df = pd.DataFrame()
        print(f"Processing subfolder: {subfolder}")
        i+=1
        print(f"{i}/{len(os.listdir(data_dir))}")
        subfolder_path = os.path.join(data_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping {subfolder_path}, not a directory.")
            continue
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.wav') and file_name.startswith('V'):
                record_path = os.path.join(subfolder_path, file_name)
                print(f"Processing {record_path}")
                record_features_df = process_audio_file(record_path, segment_duration=segment_duration)
                subfolder_features_df = pd.concat([subfolder_features_df, record_features_df], ignore_index=True)
        if not subfolder_features_df.empty:
            all_features_df = pd.concat([all_features_df, subfolder_features_df], ignore_index=True)
    if not all_features_df.empty:
        feature_columns = [col for col in all_features_df.columns if col not in ['path', 'audio_id', 'segment']]
        for col in feature_columns:
            print(f"Feature column: {col}, dtype: {all_features_df[col].dtype}")
            all_features_df[col] = all_features_df[col].astype(np.float32)
            print(f"Feature column: {col}, dtype: {all_features_df[col].dtype}")
        all_features_df = scale_features(all_features_df, feature_columns)
        all_features_df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")





def main():
    save_paths=[]
    classes = {
        os.path.join('/media', 'data', 'italian_parkinson', '22 Elderly Healthy Control'): 'Healthy',
        os.path.join('/media', 'data', 'italian_parkinson', "28 People with Parkinson's disease", '1-5'): 'Mild',
        os.path.join('/media', 'data', 'italian_parkinson', "28 People with Parkinson's disease", '6-10'): 'Mild',
        os.path.join('/media', 'data', 'italian_parkinson', "28 People with Parkinson's disease", '11-16'): 'Severe',
        os.path.join('/media', 'data', 'italian_parkinson', "28 People with Parkinson's disease", '17-28'): 'Severe',
    }
    for data_dir, label in classes.items():
        output_csv = f'/media/data/features/{label}.csv'
        preprocess_dataset(data_dir, output_csv, segment_duration=1)
        save_paths.append(output_csv)
    all_data = pd.DataFrame()
    for path in save_paths:
        df = pd.read_csv(path)
        all_data = pd.concat([all_data, df], ignore_index=True)
    all_data.to_csv('/media/data/features/extracted_features.csv', index=False)
    

if __name__ == "__main__":
    main()