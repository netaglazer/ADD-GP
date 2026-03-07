# ADD-GP

Official implementation for Few-Shot Speech Deepfake Detection Adaptation with Gaussian Processes
### Interspeech 2025
Recent advancements in Text-to-Speech (TTS) models, particularly in voice cloning, have intensified the demand for adaptable and efficient deepfake detection methods. As TTS systems continue to evolve, detection models must be able to efficiently adapt to previously unseen generation models with minimal data. This paper introduces ADD-GP, a few-shot adaptive framework based on a Gaussian Process (GP) classifier for Audio Deepfake Detection (ADD). We show how the combination of a powerful deep embedding model with the Gaussian processes flexibility can achieve strong performance and adaptability. Additionally, we show this approach can also be used for personalized detection, with greater robustness to new TTS models and one-shot adaptability. To support our evaluation, a benchmark dataset is constructed for this task using new state-of-the-art voice cloning models.
 
Link to the paper:
https://arxiv.org/abs/2505.23619

Link to LibriFake:


## 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/netaglazer/ADD-GP.git
cd ADD-GP

# Create a new conda environment
conda create -n addgp python=3.9
conda activate addgp

# Install dependencies
pip install -r requirements.txt

```

## Running ADD-GP

```bash
python3 tts_trainer.py \
--train-path  '/data/Libri_dataset_11labs_new_train.csv' 
--val-path '/data/Libri_dataset_11labs_new_val.csv' \
--test-path '/data/Libri_dataset_11labs_new_test.csv' \
--train-batch-size 70  \
--val-batch-size 230 \
--train-tts f5,yourtts,valle,Tacotron2,whisper \
--val-tts eleven_labs \
--eval True  \
--few-shot 100 \
--checkpoint-path /path/to/pretrained/xlsr/xlsr4_model_epoch_0.pt # If does not exist, loads hf pretrained
--train 1  # 0 for eval only
```
