<div align="center">
  <img src="figures/logo.png" width="300"/> 
</div>

<p align="center"><strong style="font-size: 20px;">
GenVC: Self-Supervised Zero-Shot Voice Conversion
</strong>
</p>

<p align="center">
♠︎ <a href="">Model</a>   | ♣︎ <a href="https://github.com/caizexin/GenVC">Github</a> 
|  ♥︎ <a href="https://arxiv.org/abs/2502.04519">Paper</a> | ♦︎ <a href="https://caizexin.github.io/GenVC/index.html">Demo</a>
</p>

---
GenVC is an open-source, language model-based zero-shot voice conversion system that leverages self-supervised training and supports real-time and streaming voice conversion.

<p align="center">
    <img src="figures/genVC.png" width="100%"/>
</p>


## Features

✅ **Zero-shot Voice Conversion** 

✅ **Streaming VC**

✅ **Self-supervised Training**

## Install


## Inference

### Non-streaming inference
```sh
python infer.py --model_path pre_trained/GenVC_small.pth --src_wav samples/EF4_ENG_0112_1.wav --ref_audio samples/EM1_ENG_0037_1.wav --output_path samples/converted.wav
```
### Streaming inference
```sh
python infer.py --model_path pre_trained/GenVC_small.pth --src_wav samples/EF4_ENG_0112_1.wav --ref_audio samples/EM1_ENG_0037_1.wav --output_path samples/converted.wav --streaming
```
#### Latency and RTF

#### Note

## Training
We strongly recommend using wandb.
### Dataset

```sh
CUDA_VISIBLE_DEVICES=0 python train_audio_dvae.py
```
```sh
CUDA_VISIBLE_DEVICES=0 python train_content_dvae.py
```
```sh
CUDA_VISIBLE_DEVICES=0 python train_genVC.py
```
```sh
CUDA_VISIBLE_DEVICES=0 python train_vocoder.py
```
## Future updates
☑️ **Multi-GPU Training**

☑️ **Causal Neural Vocoder**

☑️ **Multilingual VC**

## Impact Statement
While our work holds significant promise, it also carries potential societal implications that warrant consideration. GenVC is a voice conversion system capable of transforming source speech into desired voices. While this technology has valuable applications, such as enhancing privacy by anonymizing voices and enabling accessibility for individuals with speech impairments, it is also presents ethical challenges. Specifically, the ability to convincingly replicate voices can be misused to create audio deepfakes, which may be employed for malicious purposes, such as identity theft, fraud, and the spread of misinformation.

To mitigate these risks, we strongly advocate for the responsible and ethical use of voice conversion technologies. Researchers, developers, and users must comply with relevant laws and guidelines, ensuring that these systems are used exclusively for legitimate and beneficial applications. Trans-
parency, informed consent, and robust safeguards should be prioritized to prevent misuse and protect individuals’ rights and privacy.

## Acknowledgements