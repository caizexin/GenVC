import os
import random
import sys

import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from tqdm import tqdm
from utils import load_audio

torch.set_num_threads(1)

def get_prompt_slice(gt_path, max_sample_length, min_sample_length, sample_rate, is_eval=False):
    rel_clip = load_audio(gt_path, sample_rate)
    if rel_clip is None:
        return None, None
    # if eval uses a middle size sample when it is possible to be more reproducible
    
    if is_eval:
        sample_length = int((min_sample_length + max_sample_length) / 2)
    else:
        sample_length = random.randint(min_sample_length, max_sample_length)
    gap = rel_clip.shape[-1] - sample_length
    if gap < 0:
        sample_length = rel_clip.shape[-1] // 2
        gap = rel_clip.shape[-1] - sample_length

    # if eval start always from the position 0 to be more reproducible
    if is_eval:
        rand_start = 0
    else:
        rand_start = random.randint(0, gap)

    rand_end = rand_start + sample_length
    rel_clip = rel_clip[:, rand_start:rand_end]
    rel_clip = F.pad(rel_clip, pad=(0, max_sample_length - rel_clip.shape[-1]))
    # cond_idxs = [rand_start, rand_end]
    return rel_clip, rand_end - rand_start # , cond_idxs

class VCWaveDataset(torch.utils.data.Dataset):
    def __init__(self, model_args, meta_file, sample_rate, text_frame_rate, is_eval=False):
        # self.config = config
        model_args = model_args
        self.failed_samples = set()
        self.debug_failures = model_args.debug_loading_failures
        self.max_conditioning_length = model_args.max_conditioning_length
        self.min_conditioning_length = model_args.min_conditioning_length
        self.is_eval = is_eval
        self.text_frame_rate = text_frame_rate
        self.sample_rate = sample_rate
        self.max_text_len = model_args.max_text_length
        self.min_text_len = model_args.min_text_length
        self.use_masking_gt_prompt_approach = model_args.gpt_use_masking_gt_prompt_approach
        # content downsampling ratio = frame_rate * sample_rate * compression_factor, which is 4 since dvae model has a 4x compression factor
        self.content2wavRatio = int(text_frame_rate * sample_rate) * 4

        self.spk2utt = {}
        self.samples = []
        with open(meta_file) as rf:
            for line in rf:
                line = line.strip()
                parts = line.split("|")
                if len(parts) != 2:
                    print(f"Invalid line in metafile: {line}")
                    continue
                audio_file, speaker = parts
                self.samples.append({"audio_file": audio_file, "spk": speaker})
                if speaker not in self.spk2utt:
                    self.spk2utt[speaker] = []
                self.spk2utt[speaker].append(audio_file)
        
        # get valid substitution sample
        for i in range(len(self.samples)):
            audiopath = self.samples[i]["audio_file"]
            wav = load_audio(audiopath, self.sample_rate)
            if wav != None:
                self.substitution = audiopath
                break

    def load_item(self, sample):
        audiopath = sample["audio_file"]
        wav = load_audio(audiopath, self.sample_rate)
        if wav == None:
            audiopath = self.substitution
            wav = load_audio(audiopath, self.sample_rate)
            cond, cond_len = get_prompt_slice(
                audiopath, self.max_conditioning_length, 
                self.min_conditioning_length, self.sample_rate, 
                self.is_eval
            )
            ref_sample = audiopath
        else:
            if self.use_masking_gt_prompt_approach:
                cond, cond_len = get_prompt_slice(
                    audiopath, self.max_conditioning_length, 
                    self.min_conditioning_length, self.sample_rate, 
                    self.is_eval
                )
                ref_sample = audiopath
            else:
                ref_sample = np.random.choice(self.spk2utt[sample["spk"]])
                cond, cond_len = get_prompt_slice(
                    ref_sample, self.max_conditioning_length, 
                    self.min_conditioning_length, self.sample_rate, 
                    self.is_eval
                )
                if cond == None:
                    cond, cond_len = get_prompt_slice(
                        audiopath, self.max_conditioning_length, 
                        self.min_conditioning_length, self.sample_rate, 
                        self.is_eval
                    )
                    ref_sample = audiopath

        return audiopath, wav, cond, cond_len, ref_sample

    def __getitem__(self, index):
        sample = self.samples[index]
        sample_id = str(index)

        # try to load the sample, if fails added it to the failed samples list
        try:
            audiopath, wav, cond, cond_len, ref_sample = self.load_item(sample)
        except:
            if self.debug_failures:
                print(f"error loading {sample['audio_file']} {sys.exc_info()}")
            self.failed_samples.add(sample_id)
            return self[1]

        res = {
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
            "filenames": audiopath,
            "condition_path": ref_sample,
            "conditioning": cond.unsqueeze(1),
            "cond_lens": torch.tensor(cond_len, dtype=torch.long),
        }
        return res

    def __len__(self):
        return len(self.samples)
       

    def collate_fn(self, batch):
        # convert list of dicts to dict of lists
        B = len(batch)

        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        # stack for features that already have the same shape
        batch["wav_lengths"] = torch.stack(batch["wav_lengths"])
        batch["cond_lens"] = torch.stack(batch["cond_lens"])
        batch["text_lengths"] = batch["wav_lengths"] // self.content2wavRatio 
        
        # cut the condition wavs according to the shortest wavform in the batch
        # min_cond_len = torch.min(batch["cond_lens"])
        cond_len = torch.max(batch["cond_lens"])
        # if min_cond_len <= self.min_conditioning_length:
        #     cond_len = min_cond_len
        # else:
        #     max_cond_len = min(self.max_conditioning_length, min_cond_len)
        #     cond_len = random.randint(self.min_conditioning_length, max_cond_len)
        
        # randomly cut the inputs to length between self.min_text_len and self.max_text_len 
        batch_wav_len = random.randint(self.min_text_len * self.content2wavRatio, self.max_text_len * self.content2wavRatio)
        batch_wav_len = min(batch_wav_len, batch["wav_lengths"].max().item())
        batch_text_len = batch_wav_len // self.content2wavRatio
        batch_wav_len = batch_text_len * self.content2wavRatio 
        
        # create padding tensors
        wav_padded = torch.FloatTensor(B, 1, batch_wav_len)
        condition_wavs = torch.FloatTensor(B, 1, 1, cond_len)
        # initialize tensors for zero padding
        wav_padded = wav_padded.zero_()
        condition_wavs = condition_wavs.zero_()
        for i in range(B):
            # randomly cut 'text' and 'wav' to the same length 
            wav = batch["wav"][i]
            gap = wav.shape[-1] - batch_wav_len
            if gap < 0:
                new_segment_len = batch["wav_lengths"][i] // self.content2wavRatio * self.content2wavRatio
                wav = wav[:, :new_segment_len]
                wav_padded[i, :, : new_segment_len] = torch.FloatTensor(wav)
                batch["wav_lengths"][i] = new_segment_len
                batch["text_lengths"][i] = new_segment_len // self.content2wavRatio
            else:
                wav_start = random.randint(0, gap)
                wav_end = wav_start + batch_wav_len
                wav = batch["wav"][i][:, wav_start:wav_end]
                wav_padded[i, :] = torch.FloatTensor(wav)
                batch["wav_lengths"][i] = wav_end - wav_start
                batch["text_lengths"][i] = batch_text_len

            cond = batch["conditioning"][i]
            gap = cond.shape[-1] - cond_len
            assert gap >= 0
            cond_start = random.randint(0, gap)
            cond_end = cond_start + cond_len
            cond = batch["conditioning"][i][:, :, cond_start:cond_end]
            condition_wavs[i, :] = torch.FloatTensor(cond)
            batch["cond_lens"][i] = cond_len
        batch["conditioning"] = condition_wavs
        batch["wav"] = wav_padded
        return batch
    
class Waveform_DVAEDataset(torch.utils.data.Dataset):
    def __init__(self, metafile, is_eval, sample_rate=24000, max_wav_len=144000):
        self.is_eval = is_eval
        self.training_seed = 1994
        self.samples = []
        self.sample_rate = sample_rate
        self.max_wav_len = max_wav_len

        with open(metafile, "r") as f:
            for line in tqdm(f.readlines()):
                wav_path = line.strip().split('|')[0]
                self.samples.append(wav_path)
        # print the number of samples
        print(f" > Total samples in {metafile}: {len(self.samples)}")
        if not is_eval:
            random.seed(self.training_seed)
            random.shuffle(self.samples)

        # get valid substitution sample
        for i in range(len(self.samples)):
            wav = load_audio(self.samples[i], self.sample_rate)
            if wav != None:
                self.substitution = wav
                break

    def __getitem__(self, index):
        wavpath = self.samples[index]
        # D * T
        wav = load_audio(wavpath, self.sample_rate)
        # substitute nonvalid sample with the first valid sample
        if wav == None:
            wav = self.substitution
        res = {
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
        }
        return res

    def __len__(self):
        return len(self.samples)
        
    def collate_fn(self, batch):
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        batch["wav_lengths"] = torch.stack(batch["wav_lengths"])
        max_wav_len = min(batch["wav_lengths"].max(), self.max_wav_len)
        wav_padded = torch.FloatTensor(B, 1, max_wav_len)
        wav_padded = wav_padded.zero_()
        for i in range(B):
            wav = batch["wav"][i]
            gap = wav.shape[-1] - max_wav_len
            if gap < 0:
                wav_padded[i, :, : batch["wav_lengths"][i]] = wav
            else:
                start = random.randint(0, gap)
                end = start + max_wav_len
                wav_padded[i, :] = wav[:, start:end]
                batch["wav_lengths"][i] = max_wav_len
        batch["wav"] = wav_padded
        return batch

