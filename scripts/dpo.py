import pandas as pd
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from tqdm import tqdm
import torch
import argparse
import re
import json
from collections import Counter
import wandb
import datetime
from transformers import (
  AutoModelForCausalLM,
  AutoModelForSequenceClassification, 
  AutoTokenizer, 
  logging, 
  EarlyStoppingCallback
)
from trl import (
  DPOConfig,
  DPOTrainer,
  create_reference_model
)
from datasets import Dataset, DatasetDict
import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import gen_pipeline, evaluate

def main():
    gen_peft_config = LoraConfig(
        r=16,  # LoRAアテンションの次元
        lora_alpha=16,  # LoRAスケーリングのAlphaパラメータ
        lora_dropout=0.1,  # LoRA レイヤーのドロップアウト確率
        bias="none",  # LoRAのバイアス種別 ("none","all", "lora_only")
        task_type="CAUSAL_LM",  # タスク種別
        target_modules=["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "k_proj", "v_proj"],
    )
    # 学習モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        config["target_model"], 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
        )
        
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 32,
        "min_new_tokens":4,
        "repetition_penalty":1.3,
        "no_repeat_ngram_size":2,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "num_return_sequences":2
    }

    
    if args.do_preprocess:
        model.cuda()
        def extract_dialogue(text: str) -> str:
            # 1) 会話ブロックを抽出
            m = re.search(r"会話履歴：\s*(.+?)\s*^###\s*応答:", text, flags=re.DOTALL | re.MULTILINE)
            if not m:
                return ""
            block = m.group(1)

            # 2) 対話行だけ抽出（先頭がシステム: or ユーザー:）
            lines = block.splitlines()
            kept = [ln for ln in lines if re.match(r"^\s*(システム|ユーザー)\s*:\s*.*", ln)]
            return "\n".join(kept)


        # 評価モデルの読み込み
        eval_peft_config = PeftConfig.from_pretrained(config["eval_model"])
        eval_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=eval_peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            num_labels=1,
            return_dict=True,
            )
        eval_model = PeftModel.from_pretrained(eval_model, config["eval_model"])
        eval_tokenizer = AutoTokenizer.from_pretrained(eval_peft_config.base_model_name_or_path, legacy=False)
        eval_model.cuda()
        
        # 学習データの処理
        df = pd.read_csv(config["train_data_path"])
        df = df[["text"]]
        df["context"] = df["text"].map(extract_dialogue)
        df["context"] += "\nシステム: "
        df[["r1", "r2"]] = df["context"].progress_map(lambda x:gen_pipeline(prompt=x)).tolist()


        df[f"aif_r1"] = df.progress_apply(lambda x:evaluate(prompt=x["context"]+x["r1"], eval_model=eval_model, eval_tokenizer=eval_tokenizer), axis=1) 
        df[f"aif_r2"] = df.progress_apply(lambda x:evaluate(prompt=x["context"]+x["r2"], eval_model=eval_model, eval_tokenizer=eval_tokenizer), axis=1)
        df["chosen"] = df.progress_apply(lambda row: row["r1"] if row[f"aif_r1"]>row[f"aif_r2"] else row["r2"], axis=1) 
        df["rejected"] = df.progress_apply(lambda row: row["r2"] if row[f"aif_r1"]>row[f"aif_r2"] else row["r1"], axis=1) 
        df.rename(columns={"context":"prompt"}, inplace=True)
        # DPOの処理
        if not os.path.exists(config["output_dir"]):
            os.makedirs(config["output_dir"])

    if args.do_train:
        def build_dpo_dataset(df):
            df = df.astype(str)
            prompts = df["prompt"].tolist()
            rejects = df["rejected"].tolist()
            chosens = df["chosen"].tolist()
            dataset = DatasetDict({"train": Dataset.from_dict({"prompt": prompts, "rejected": rejects, "chosen": chosens})})
            return dataset["train"]


        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        wandb.init(
            project = "dpo",
            name = model.config._name_or_path.replace("/","-")+now.date().strftime('%Y-%m-%d'),
            group = "dpo"
        )          
        print ("preparing lora...")
        model.config.use_cache = False
        model = get_peft_model(model, gen_peft_config)
        model_ref = create_reference_model(model)
        
        print ("load data...")
        df = pd.read_csv(config["output_dir"]+f"/all.csv")

        
        print ("build dpo dataset...")
        ds = build_dpo_dataset(df)
        splited = ds.train_test_split(test_size=0.3, shuffle=True)
        train_dataset = splited["train"]
        eval_dataset = splited["test"]
        output_dir = f"{config['output_dir']}/dpo/{model.config._name_or_path.replace('/', '-')}"
        train_dataset.save_to_disk(f"{output_dir}/train")
        eval_dataset.save_to_disk(f"{output_dir}/eval")
        print ("output_dir:", output_dir)

        print ("dpo config...")
        training_args = DPOConfig(
            output_dir= output_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            eval_strategy=config["eval_strategy"],
            per_device_train_batch_size = config["per_device_train_batch_size"],
            per_device_eval_batch_size  = config["per_device_eval_batch_size"],
            gradient_accumulation_steps = config["gradient_accumulation_steps"],
            logging_dir=output_dir + "/logs",
            num_train_epochs=config["num_train_epochs"],
            dataloader_pin_memory=False,
            save_strategy=config["save_strategy"],
            save_steps=config["save_steps"],
            save_total_limit=config["save_total_limit"],
            logging_steps=config["logging_steps"],
            learning_rate=config["learning_rate"],
            lr_scheduler_type= config["lr_scheduler_type"],
            warmup_steps=config["warmup_steps"],
            beta=config["beta"],
            bf16=config["bf16"],
            max_prompt_length=config["max_prompt_length"],
            max_length=config["max_length"],
            load_best_model_at_end=True,
            remove_unused_columns=True,
            metric_for_best_model="eval_loss",
            report_to=config["report_to"],
        )
        
        dpo_trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )
        print ("training...")
        dpo_trainer.train()
        dpo_trainer.save_model(f"{output_dir}/best")
        train_dataset.save_to_disk(f"{output_dir}/JEmpathetic-train")
        eval_dataset.save_to_disk(f"{output_dir}/JEmpathetic-eval")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DPOで学習するプログラム')
    parser.add_argument('--config', help='config.jsonのパス', default="config/rlaif_1.8b.json")  
    parser.add_argument('--do_preprocess', help='前処理を実行するかどうか', action='store_true')
    parser.add_argument('--do_train', help='学習を実行するかどうか', action='store_true')
    args = parser.parse_args()
    tqdm.pandas()
    config = json.load(open(args.config))
    main()