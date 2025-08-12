#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
from transformers import AutoTokenizer,  AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import torch
import argparse
import os
from openai import OpenAI
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def main():
    # ベースモデルの読み込み
    peft_config = PeftConfig.from_pretrained(args.eval_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        num_labels=1,
        return_dict=True,
        )
    model = PeftModel.from_pretrained(model, args.eval_model)
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    def evaluate(prompt):
        with torch.no_grad():
            encoded = tokenizer.encode(prompt, return_tensors="pt").cuda()
            logits = model(encoded).logits
        return logits[0][0].item()

    

    def is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False
    def gpt3(prompt):
        # OpenAI APIを使用してgpt-3.5-turboモデルで応答を評価する関数
        chat_completion = " "
        while not is_number(chat_completion[0]):
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": df_eval["text"][0]}],
                max_tokens=4
            ).choices[0].message.content
        return chat_completion[0]
    
    df_eval = pd.read_csv(args.eval_data_path)
    df_eval["zeroshot"] = df_eval["text"].progress_map(gpt3)
    df_eval["zeroshot"]

    df_eval["wsft"] = df_eval["text"].progress_apply(evaluate)
    df_eval["wsft"]

    print(" ---相関係数の計算・w/SFT(spearman, pearson)---")
    print(round(df_eval[["label", "wsft"]].corr("spearman").iloc[1, 0], 3), round(df_eval[["label", "wsft"]].corr("pearson").iloc[1, 0], 3))
    print(" ---相関係数の計算・chatgpt(spearman, pearson)---")
    print(round(df_eval[["label", "zeroshot"]].corr("spearman").iloc[1, 0], 3), round(df_eval[["label", "zeroshot"]].corr("pearson").iloc[1, 0], 3))
    df_eval.to_csv(args.output_csv_path, index=False, encoding="utf-8-sig")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewardモデルの評価をするプログラム')
    parser.add_argument('--eval_model', help='aifモデルのパス', default="checkpoints/mlp/final_model")    # 必須の引数を追加
    parser.add_argument("--eval_data_path", help="評価データのパス", default="checkpoints/mlp/test.csv")
    parser.add_argument("--output_csv_path", help="評価結果の出力先", default="outputs/eval_reward_model.csv")
    args = parser.parse_args([])
    tqdm.pandas()
    
    main()
