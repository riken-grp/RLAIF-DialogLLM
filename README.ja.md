[English](README.md) | [日本語](README.ja.md)

これは日本語版のREADMEです。

# RLAIF-RealPersonaChat: Real Persona Chatを用いたRLAIF実装

このリポジトリでは、[Training Dialogue Systems by AI Feedback for Improving Overall Dialogue Impression](https://ieeexplore.ieee.org/document/10888775)のDirect Preference Optimizationの再現実装を提供します。

ただし、学習データの権利関係から、本リポジトリは学習データとして論文内とは異なる[Real Persona Chat](https://github.com/nu-dialogue/real-persona-chat)を用いる構成になっています。

## 📝 プロジェクト概要

本プロジェクトでは、Real Persona Chatデータセットの「情報量」評価指標を用いて、LLMを用いた回帰モデルの学習とそれを用いた対話モデルのDirect Preference Optimization(DPO)を行うためのコードを提供します。：

- **Real Persona Chat**: nu-dialogue/real-persona-chatデータセット使用
- **評価指標**: informativeness, comprehension, familiarity, interest, proactiveness, satisfaction
- **RLAIF**: AI Feedbackを用いた報酬モデル学習とDPO最適化
- **報酬モデルとして用いるベースモデル**: llm-jp/llm-jp-3-1.8b-instruct
- **生成モデルとして用いるベースモデル**: sbintuitions/sarashina2.2-1b

## 🏗️ プロジェクト構造

```
rlaif-realpersonachat/
├── scripts/                     # 実行スクリプト
│   ├── format_realpersonachat.py    # データセットのダウンロードと前処理
│   ├── train_evaluator_rpc.py       # 報酬モデル(LLMを用いた回帰モデル)の学習
│   ├── evaluate_reward_model.py     # 報酬モデルの評価
│   ├── dpo.py                       # 報酬モデルを用いたDPOによる対話モデル学習
│   ├── evaluate_dialog_model.py     # 対話モデル評価
│   ├── utils/
│   │   └── utils.py                 # ユーティリティ関数
├── config/                      # 設定ファイル
│   ├── evaluator_1.8b.json         # 評価モデル学習設定
│   └── rlaif_1.8b.json             # DPO学習設定
├── resources/                   
│   ├── instruction_informativeness.txt # 報酬モデル用の評価プロンプトテンプレート
├── checkpoints/                 # 学習済みモデル
├── outputs/                     # 実験結果
```

## 🚀 実行手順

以下の順序で実行してください：

### 1. データセット前処理 (format_realpersonachat)

Real Persona Chatデータセットをダウンロードし、学習用フォーマットに変換します：

```bash
python scripts/format_realpersonachat.py --label informativeness
```

**オプション**:
- `--label`: 対象評価指標 (informativeness, comprehension, familiarity, interest, proactiveness, satisfaction)、デフォルトはinformativeness

**出力**: `resources/real_persona_chat.csv` (約40万件の対話データ)

### 2. 評価モデル学習 (train_evaluator)

情報量を評価する報酬モデルを学習します：

```bash
python scripts/train_evaluator_rpc.py --config config/evaluator_1.8b.json
```



**出力**: `checkpoints/mlp/` に学習済み評価モデル

**メモリ要件**: 
- 通常時: 約20GB VRAM
- 学習時: 一時的に最大23.5GB VRAM使用
- RTX 3090で約40分程度

### 3. 報酬モデル評価 (evaluate_reward_model)

学習した評価モデルの性能を確認します：

```bash
# 実行コマンド
python scripts/evaluate_reward_model.py --eval_model checkpoints/mlp/checkpoint-XXXX

# 出力例
報酬モデル
 ---相関係数の計算・w/SFT(spearman, pearson)---
0.297 0.292
 ---相関係数の計算・chatgpt(spearman, pearson)---
-0.023 0.002
```

**機能**:
- 学習済み評価モデルの性能検証
- GPT-3.5を用いたzero-shot promptingと対象モデルの比較評価
- 評価結果の出力

### 4. DPO学習

評価モデルを用いたDPO用学習データの構築、およびDPOによる対話モデルの学習をします：

```bash
python scripts/dpo.py --config config/rlaif_1.8b.json --do_preprocess --do_train
```

**処理内容**:
- 対話データから2つの応答を生成
- 評価モデルで応答を評価
- 高評価応答をchosen、低評価応答をrejectedとしてDPO学習

**出力**: `checkpoints/dpo/` に改善されたモデル
**メモリ要件**: 
- 一時的に最大23.5GB VRAM使用
- RTX 3090で約30分ほど

### 5. 対話モデル評価 (evaluate_dialog_model)

DPO前後のモデル性能を比較評価します：

```bash
# ベースモデルの評価
python scripts/evaluate_dialog_model.py --dialog_model sbintuitions/sarashina2.2-1b

# DPO適用後モデルの評価
python scripts/evaluate_dialog_model.py --dialog_model checkpoints/dpo/checkpoint-XXXX
```

**評価結果例**:
```
DPO適用前: 4.791781556372549
DPO適用後: 4.852519914215686
```

## 🔧 設定ファイル詳細

### evaluator_1.8b.json (評価モデル学習設定)

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

### rlaif_1.8b.json (DPO学習設定)

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

## 📋 依存パッケージ

```bash
pip install torch transformers datasets trl peft wandb pandas scikit-learn tqdm openai
```
torchは各自のcudaに合ったversionを[公式](https://pytorch.org/get-started/locally/)からインストール

### メモリ不足の場合

```bash
# batch_sizeを減らす
--per_device_train_batch_size 2

# MAX_LENを調整
--max_length 512
```

### 学習が遅い場合

```bash
# gradient_accumulation_stepsを増やしてbatch_sizeを調整
--gradient_accumulation_steps 8
--per_device_train_batch_size 1
```

## 📊 ログとモニタリング

### WandB設定

```bash
wandb login
export WANDB_PROJECT="rlaif-realpersonachat"
```

### ログの確認

```bash
# 学習ログ
tail -f logs/training.log

# WandBダッシュボード
wandb sync wandb/
```

### 環境変数の設定
OPEN AIのape keyが報酬モデルの比較評価のために必要です(必要ない場合はevaluate_reward_model.pyの該当箇所を削除してください)
```
export OPENAI_API_KEY="your open ai api key"
```

## 🤝 使用方法（クイックスタート）

```bash
# 1. データ準備
python scripts/format_realpersonachat.py --label informativeness

# 2. 評価モデル学習
python scripts/train_evaluator_rpc.py --config config/evaluator_1.8b.json

# 3. 評価モデル性能確認
python scripts/evaluate_reward_model.py --eval_model checkpoints/mlp/checkpoint-6795

# 4. DPO学習
python scripts/dpo.py --config config/rlaif_1.8b.json --do_preprocess

# 5. 最終評価
python scripts/evaluate_dialog_model.py --dialog_model checkpoints/dpo/final
```

