import os
from glob import glob
import numpy as np
import shutil
import torch
import torchaudio 
import torchaudio.transforms as T

target_sample_rate = 16000
n_fft = 2048
hop_length = 512
targets = [1,2,3,4,5,6,7]

path1 = "./data/22 Elderly Healthy Control"
path2 = "./data/28 People with Parkinson's disease"

def healthy_control(path):
    for subfolder in os.listdir(path):
        if os.path.isdir(os.path.join(path, subfolder)):
            print(f"Processing folder: {subfolder}")
            for file in os.listdir(os.path.join(path, subfolder)):
                if file.endswith(".wav") and file.startswith("V"):
                    file_path = os.path.join(path, subfolder, file)
                    waveform, sample_rate = torchaudio.load(file_path)
                    
                    # Resample if needed
                    if sample_rate != target_sample_rate:
                        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                        waveform = resampler(waveform)
                        sample_rate = target_sample_rate
                    #trim remove silence in beginning and end
                    waveform = T.Vad(sample_rate=sample_rate)(waveform)
                    
                    for target in targets:
                        target_duration = target * target_sample_rate
                        step = target_duration // 2  # 50% overlap

                        num_segments = max(1, (waveform.size(1) - target_duration) // step + 1)

                        for i in range(num_segments):
                            start = i * step
                            end = start + target_duration
                            segment = waveform[:, start:end]

                            # Pad if segment is shorter
                            if segment.size(1) < target_duration:
                                pad_amount = target_duration - segment.size(1)
                                segment = torch.nn.functional.pad(segment, (0, pad_amount))
                            elif segment.size(1) > target_duration:
                                segment = segment[:, :target_duration]

                            # Save segment
                            base_dir = f"{target}S/{target}AS/healthy"
                            os.makedirs(base_dir, exist_ok=True)
                            segment_path = os.path.join(base_dir, f"{file[:-4]}_seg{i}.wav")
                            torchaudio.save(segment_path, segment, target_sample_rate)

                            # Save first segment only to FS
                            if i == 0:
                                fs_dir = f"{target}S/{target}FS/healthy"
                                os.makedirs(fs_dir, exist_ok=True)
                                fs_path = os.path.join(fs_dir, f"{file[:-4]}_seg0.wav")
                                torchaudio.save(fs_path, segment, target_sample_rate)



def parkinsons_disease(path):
    for subfolder in os.listdir(path):
        class_name = ""
        if subfolder=="1-5" or subfolder=="6-10":
            class_name="mild"
        elif subfolder=="11-16" or subfolder=="17-28":
            class_name="severe"
        for sub_subfolder in os.listdir(os.path.join(path, subfolder)):
            if os.path.isdir(os.path.join(path, subfolder, sub_subfolder)):
                print(f"Processing folder: {sub_subfolder}")
                for file in os.listdir(os.path.join(path, subfolder, sub_subfolder)):
                    if file.endswith(".wav") and file.startswith("V"):
                        file_path = os.path.join(path, subfolder, sub_subfolder, file)
                        waveform, sample_rate = torchaudio.load(file_path)
                        
                            # Resample if needed
                        if sample_rate != target_sample_rate:
                            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                            waveform = resampler(waveform)
                            sample_rate = target_sample_rate
                        #trim remove silence in beginning and end
                        waveform = T.Vad(sample_rate=sample_rate)(waveform)
                        for target in targets:
                            target_duration = target * target_sample_rate
                            step = target_duration // 2  # 50% overlap

                            num_segments = max(1, (waveform.size(1) - target_duration) // step + 1)

                            for i in range(num_segments):
                                start = i * step
                                end = start + target_duration
                                segment = waveform[:, start:end]
                                if segment.size(1) < target_duration:
                                    pad_amount = target_duration - segment.size(1)
                                    segment = torch.nn.functional.pad(segment, (0, pad_amount))
                                elif segment.size(1) > target_duration:
                                    segment = segment[:, :target_duration]
                                
                                
                                # Save segment
                                base_dir = f"{target}S/{target}AS/{class_name}"     
                                os.makedirs(base_dir, exist_ok=True)
                                segment_path = os.path.join(base_dir, f"{file[:-4]}_seg{i}.wav")
                                torchaudio.save(segment_path, segment, target_sample_rate)

                                # Save first segment only to FS
                                if i == 0:
                                    fs_dir = f"{target}S/{target}FS/{class_name}"
                                    os.makedirs(fs_dir, exist_ok=True)
                                    fs_path = os.path.join(fs_dir, f"{file[:-4]}_seg0.wav")
                                    torchaudio.save(fs_path, segment, target_sample_rate)

            
healthy_control(path1)
parkinsons_disease(path2)