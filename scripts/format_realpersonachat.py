import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
tqdm.pandas()

def main():
    dialogue_dataset = load_dataset(
            "nu-dialogue/real-persona-chat",
            name="dialogue",
            trust_remote_code=True
        )
    # Convert datasets to pandas DataFrames
    df_dialogue = dialogue_dataset["train"].to_pandas()
    def format_dialog(d:dict):
        interlocutor_ids = d["interlocutor_id"]
        utterancs = d["text"]
        dialog = "\n".join([f"{i}: {u}" for i, u in zip(interlocutor_ids, utterancs)])
        return     [
                        dialog.replace(interlocutor_ids[0], "システム").replace(interlocutor_ids[1], "ユーザー"),
                        dialog.replace(interlocutor_ids[0], "ユーザー").replace(interlocutor_ids[1], "システム")
                    ]
    print(format_dialog(df_dialogue["utterances"][0])[0])

    def format_dataframe(d_evaluations, d_utterances):
        df = pd.DataFrame(d_evaluations)
        df["utterances"] = format_dialog(d_utterances)
        # 積極性だけ自身で自身を評価している指標なので、ユーザーとシステムの評価を入れ替える
        df.loc[[0, 1], "proactiveness"] = df.loc[[1, 0], "proactiveness"].values
        return df.iloc[0, :]
    idx = 5
    format_dataframe(df_dialogue.iloc[idx]["evaluations"], df_dialogue.iloc[idx]["utterances"])
    
    # Apply the formatting function to each row of the DataFrame
    df = df_dialogue.progress_apply(lambda x:format_dataframe(x["evaluations"], x["utterances"]), axis=1)
    df = df.reset_index(drop=True)
    print(df.head(1))
    
    df.rename(columns={
        "utterances": "text",
        args.label: "label"
    }, inplace=True)
    
    df[["text", "label"]].to_csv("resources/real_persona_chat.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format Real Persona Chat dataset")
    parser.add_argument("--label", default="informativeness", choices=["informativeness", "comprehension", "familiarity", "interest", "proactiveness", "satisfaction"], help="Target evaluation metric to format")
    args = parser.parse_args()
    main()