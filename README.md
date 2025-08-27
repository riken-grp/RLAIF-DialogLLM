[English](README.md) | [日本語](README.ja.md)

This is the English README.

# RLAIF-RealPersonaChat: RLAIF Implementation using Real Persona Chat

This repository provides a reproduction of Direct Preference Optimization (DPO) from [Training Dialogue Systems by AI Feedback for Improving Overall Dialogue Impression](https://ieeexplore.ieee.org/document/10888775).

Due to dataset licensing considerations, this repository uses [Real Persona Chat](https://github.com/nu-dialogue/real-persona-chat) as training data instead of the dataset used in the paper.

## 📝 Project Overview

This project provides code to (1) train a regression-based reward model using the **informativeness** metric from the Real Persona Chat dataset, and (2) apply Direct Preference Optimization (DPO) to a dialogue model using that reward model.

- **Dataset**: nu-dialogue/real-persona-chat  
- **Evaluation metrics**: informativeness, comprehension, familiarity, interest, proactiveness, satisfaction  
- **RLAIF**: Reward model training with AI Feedback and DPO optimization  
- **Base model for the reward model**: llm-jp/llm-jp-3-1.8b-instruct  

## 🏗️ Project Structure



```
rlaif-realpersonachat/
├── scripts/
│ ├── format_realpersonachat.py # Download and preprocess the dataset
│ ├── train_evaluator_rpc.py # Train reward model (regression with an LLM)
│ ├── evaluate_reward_model.py # Evaluate the reward model
│ ├── dpo.py # Train dialogue model with DPO using the reward model
│ ├── evaluate_dialog_model.py # Evaluate dialogue models
│ ├── utils/
│ │ └── utils.py # Utility functions
├── config/
│ ├── evaluator_1.8b.json # Config for reward model training
│ └── rlaif_1.8b.json # Config for DPO training
├── resources/
│ ├── instruction_informativeness.txt # Evaluation prompt template for the reward model
├── checkpoints/ # Trained models
├── outputs/ # Experiment results
```

## 🚀 Execution Steps

Please follow the steps below in order:

### 1. Dataset Preprocessing (format_realpersonachat)

Download the Real Persona Chat dataset and convert it into the training format:

```bash
python scripts/format_realpersonachat.py --label informativeness
```

**Options**:
- `--label`: Target evaluation metric (informativeness, comprehension, familiarity, interest, proactiveness, satisfaction).Default: informativeness.

**Output**: `resources/real_persona_chat.csv` (approximately 400,000 dialogue samples)

### 2. Train the Reward Model (train_evaluator)

Train the reward model to evaluate **informativeness**:

```bash
python scripts/train_evaluator_rpc.py --config config/evaluator_1.8b.json
```


**Output**: Trained reward model stored in `checkpoints/mlp/`

**Memory Requirements**:  
- Typical: ~20 GB VRAM  
- Peak during training: up to ~23.5 GB VRAM  
- On RTX 3090: approximately 40 minutes

### 3. Evaluate the Reward Model (evaluate_reward_model)

Check the performance of the trained reward model:

```bash
# Command
python scripts/evaluate_reward_model.py --eval_model checkpoints/mlp/checkpoint-XXXX

# Example output
Reward Model
 ---Correlation (w/SFT; Spearman, Pearson)---
0.297  0.292
 ---Correlation (ChatGPT; Spearman, Pearson)---
-0.023  0.002
```
**Functions**:
- Validate the performance of the trained reward model  
- Compare against GPT-3.5 using zero-shot prompting  
- Output evaluation results  

### 4. DPO Training

Construct training data for DPO using the reward model and train a dialogue model with DPO:

```bash
python scripts/dpo.py --config config/rlaif_1.8b.json --do_preprocess --do_train
```
**Process**:
- Generate two responses from dialogue data
- Evaluate the responses with the reward model
- Label the higher-scored response as chosen and the lower-scored response as rejected, then train with DPO

**Output**: Improved model stored in `checkpoints/dpo/` 

**Memory Requirements**: 
- Up to ~23.5 GB VRAM (peak)
- On RTX 3090: approximately 30 minutes

### 5. Dialogue Model Evaluation (evaluate_dialog_model)

Compare the performance of the dialogue model before and after DPO:

```bash
# Evaluate the base model
python scripts/evaluate_dialog_model.py --dialog_model sbintuitions/sarashina2.2-1b

# Evaluate the model after applying DPO
python scripts/evaluate_dialog_model.py --dialog_model checkpoints/dpo/checkpoint-XXXX
```

**Example Results**:
```
Before DPO: 4.791781556372549
After DPO: 4.852519914215686
```

## 🔧 Config File Details

### evaluator_1.8b.json (Reward Model Training Settings)

```json
{
    "model": "llm-jp/llm-jp-3-1.8b-instruct",
    "train_data_path": "./resources/real_persona_chat.csv",
    "instruction_path": "./resources/instruction_informativeness.txt",
    "output_dir": "./checkpoints/mlp",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 1e-4,
    "lora_r": 64,
    "lora_alpha": 16
}
```

### rlaif_1.8b.json (DPO Training Settings)

```json
{
    "target_model": "sbintuitions/sarashina2.2-1b",
    "eval_model": "checkpoints/mlp/checkpoint-6795",
    "output_dir": "checkpoints/dpo",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 2,
    "learning_rate": 1e-5,
    "beta": 0.25
}
```

## 📋 Dependencies
```bash
pip install torch transformers datasets trl peft wandb pandas scikit-learn tqdm openai
```
Install `torch` with the version that matches your CUDA environment from the [official site](https://pytorch.org/get-started/locally/).

### In case of Out-of-Memory (OOM)

```bash
# Reduce batch size
--per_device_train_batch_size 2

# Adjust MAX_LEN
--max_length 512
```

## If Training is Slow
```bash
# Increase gradient_accumulation_steps and adjust batch size
--gradient_accumulation_steps 8
--per_device_train_batch_size 1
```

## 📊 Logging and Monitoring
### WandB Setup
```bash
wandb login
export WANDB_PROJECT="rlaif-realpersonachat"
```

```bash
# Training logs
tail -f logs/training.log

# WandB dashboard
wandb sync wandb/
```

### Environment Variables

An **OpenAI API key** is required for comparative evaluation with the reward model.  
(If not needed, you can remove the corresponding part in `evaluate_reward_model.py`.)

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 🤝 Usage (Quick Start)

```bash
# 1. Data preparation
python scripts/format_realpersonachat.py --label informativeness

# 2. Train the reward model
python scripts/train_evaluator_rpc.py --config config/evaluator_1.8b.json

# 3. Check reward model performance
python scripts/evaluate_reward_model.py --eval_model checkpoints/mlp/checkpoint-6795

# 4. DPO training
python scripts/dpo.py --config config/rlaif_1.8b.json --do_preprocess

# 5. Final evaluation
python scripts/evaluate_dialog_model.py --dialog_model checkpoints/dpo/final
```

## Citation
```
@inproceedings{yoshida-etal-2025-aif,
  author={Yoshida, Kai and Mizukami, Masahiro and Kawano, Seiya and Kruengkrai, Canasai and Sugiyama, Hiroaki and Yoshino, Koichiro},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Training Dialogue Systems by AI Feedback for Improving Overall Dialogue Impression}, 
  year={2025},
  pages={1-5},
  keywords={Measurement;Training;Hands;Adaptation models;Large language models;Reinforcement learning;Oral communication;Signal processing;Speech processing;Tuning;Dialogue System;Conversation System;RLAIF;RLHF;LLM},
  doi={10.1109/ICASSP49660.2025.10888775}}
```
If you have any questions about the paper and repository, feel free to contact Kai Yoshida (yoshida.kai.yf1 [at] is.naist.jp) or open an issue.