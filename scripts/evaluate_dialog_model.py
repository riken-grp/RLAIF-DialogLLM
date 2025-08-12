#!/usr/bin/env python
# coding: utf-8


import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM,  AutoModelForSequenceClassification
from datasets import load_from_disk
import pandas as pd
import argparse
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import os,sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import gen_pipeline, evaluate



def main():
    # 評価モデルの読み込み
    peft_config = PeftConfig.from_pretrained(args.eval_model)
    eval_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        num_labels=1,
        return_dict=True,
        )
    eval_model = PeftModel.from_pretrained(eval_model, args.eval_model)
    eval_tokenizer = AutoTokenizer.from_pretrained(eval_model.config._name_or_path, legacy=False)
    eval_model.cuda()
    
    # 生成モデルの読み込み 
    peft_config = LoraConfig(
        r=16,  # LoRAアテンションの次元
        lora_alpha=16,  # LoRAスケーリングのAlphaパラメータ
        lora_dropout=0.1,  # LoRA レイヤーのドロップアウト確率
        bias="none",  # LoRAのバイアス種別 ("none","all", "lora_only")
        task_type="CAUSAL_LM",  # タスク種別
        target_modules=["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "k_proj", "v_proj"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.dialog_model, 
        torch_dtype=torch.bfloat16,
        )
    if "checkpoint" in args.dialog_model:
        model = PeftModel.from_pretrained(model, args.dialog_model)
    else:
        model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(args.dialog_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.cuda()
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 32,
        "min_new_tokens":4,
        "repetition_penalty":1.3,
        "no_repeat_ngram_size":2,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # 学習データの処理
    ds = load_from_disk(args.eval_data_path)
    df = pd.DataFrame(ds)
    df.rename(columns={"prompt":"context"}, inplace=True)
    df["responces"] = df["context"].progress_map(lambda x:gen_pipeline(prompt=x, tokenizer=tokenizer, model=model, generation_kwargs=generation_kwargs)[0])

    
    df["aif"] = df.progress_apply(lambda x:evaluate(x["context"]+x["responces"], tokenizer=eval_tokenizer, eval_model=eval_model), axis=1)
    print("----aif----", df["aif"].mean())
    df.to_csv(args.output_file_name, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NTTのデータでAIFをSFTプログラム')
    parser.add_argument("--eval_model", default="checkpoints/mlp/final_model", type=str)
    parser.add_argument("--dialog_model", default="checkpoints/dpo/dpo/sbintuitions-sarashina2.2-1b/best", type=str)
    parser.add_argument("--eval_data_path", default="checkpoints/dpo/dpo/sbintuitions-sarashina2.2-1b/eval", type=str)
    parser.add_argument("--output_file_name", default="outputs/eval_dialog_model.csv", type=str)
    args = parser.parse_args()
    tqdm.pandas()
    main()