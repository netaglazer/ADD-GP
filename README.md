# ADD-GP

Official implementation for Few-Shot Speech Deepfake Detection Adaptation with Gaussian Processes
 
Link to the paper:

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