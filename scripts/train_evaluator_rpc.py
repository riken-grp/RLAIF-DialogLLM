#!/usr/bin/env python
# coding: utf-8
# a
# In[2]:


"""
batch_size=2のloraでswallow7bに40GBぐらいつかう
"""


# In[3]:


"""
watch -n 1 -d -t nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_evaluator.py
"""


# In[6]:


import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm
import torch
import json
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
import argparse
import evaluate
from datasets import Dataset, Value



def main():
    # Load Model and Tokenizer
    print(f"Loading model {config['model']}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"], 
        num_labels=1,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
        )
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Preparing datasets...")
    # Load Prompt Template
    with open(config["instruction_path"], "r") as f:
        prompt_template = f.read() 
    
    # Split dataset into train, eval, valid
    df = pd.read_csv(config["train_data_path"])
    df["text"] = df.progress_apply(lambda x: prompt_template.format(source=x["text"]), axis=1)
    print(type(df["label"][0]))
    df_train, df_eval = train_test_split(df, test_size=0.2, random_state=42)
    df_eval, df_valid = train_test_split(df_eval, test_size=0.5, random_state=42)
    print(df_train.reset_index(drop=True).head())
    print(len(df), len(df_train), len(df_eval), len(df_valid))

    
    df_train.to_csv(f"{config['output_dir']}/train.csv", index=False, encoding="utf-8-sig")
    df_eval.to_csv(f"{config['output_dir']}/test.csv", index=False, encoding="utf-8-sig")
    df_valid.to_csv(f"{config['output_dir']}/valid.csv", index=False, encoding="utf-8-sig")
    train_dataset = Dataset.from_dict({"text": df_train["text"].tolist(), "label": df_train["label"].tolist()})
    valid_dataset = Dataset.from_dict({"text": df_valid["text"].tolist(), "label": df_valid["label"].tolist()})
    # label列を float32 にキャスト
    train_dataset = train_dataset.cast_column("label", Value("float32"))
    valid_dataset = valid_dataset.cast_column("label", Value("float32"))

    MAX_LEN = 512
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    train_dataset.set_format("torch")
    valid_dataset = valid_dataset.map(tokenize, batched=True, remove_columns=["text"])
    valid_dataset.set_format("torch")
    
    
    
    # 学習の設定
    metric = evaluate.load("mse")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(
        output_dir = config["output_dir"],
        num_train_epochs = config["num_train_epochs"],
        per_device_train_batch_size = config["per_device_train_batch_size"],
        logging_dir = "./logs",
        logging_steps = config["logging_steps"],
        save_steps = config["save_steps"],
        save_strategy = config["save_strategy"],
        eval_strategy = config["eval_strategy"],
        eval_steps = config["eval_steps"],
        save_total_limit = config["save_total_limit"],
        learning_rate = config["learning_rate"],
        warmup_steps = config["warmup_steps"],
        weight_decay = config["weight_decay"],
        gradient_accumulation_steps = config["gradient_accumulation_steps"],
        remove_unused_columns = True,
        bf16 = True,
        group_by_length=True, 
        overwrite_output_dir = True,
        run_name =f"aif_sft_{config['model'].replace('/', '_')}_rpc",
        report_to = config["report_to"],
    )
    peft_config = LoraConfig(
        r=config["lora_r"],  # LoRAアテンションの次元
        lora_alpha=config["lora_alpha"],  # LoRAスケーリングのAlphaパラメータ
        lora_dropout=config["lora_dropout"],  # LoRA レイヤーのドロップアウト確率
        bias=config["lora_bias"],  # LoRAのバイアス種別 ("none","all", "lora_only")
        task_type=TaskType.SEQ_CLS,  # タスク種別
        target_modules=config["target_modules"],  # LoRAを適用するモジュール
        modules_to_save=["score"]
    )



    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    #print(model.eval())
    #print(model.base_model.model.score.original_module.weight.requires_grad)
    torch.save(model.score.state_dict(), f"{config['output_dir']}/score.bin")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest")
    )
    trainer.train()
    trainer.model.save_pretrained(f"{config['output_dir']}/final_model")
    model.config.to_json_file(f"{config['output_dir']}/final_model/config.json")
    torch.save(model.state_dict(), f"{config['output_dir']}/final_model/model.bin")
    model.save_pretrained(f"{config['output_dir']}/lora_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train evaluator(LLM+MLP) by RealPersonaChat dataset')
    parser.add_argument('--config', help='config.jsonのパス')    # 必須の引数を追加
    args = parser.parse_args()
    tqdm.pandas()
    config = json.load(open(args.config))
    main()
