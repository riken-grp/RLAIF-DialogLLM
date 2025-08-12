import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

def evaluate(prompt:str, tokenizer:AutoTokenizer, eval_model:AutoModelForSequenceClassification) -> float:
    with torch.no_grad():
        encoded = tokenizer.encode(prompt, return_tensors="pt")
        logits = eval_model(encoded.to(eval_model.device)).logits
    return logits[0][0].item()


def gen_pipeline(prompt:str, model, tokenizer:AutoTokenizer, generation_kwargs:dict) -> list[str]: 
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    )
    with torch.no_grad():
        tokens = model.generate(
            input_ids.to(device=model.device),
            **generation_kwargs
        )
    tokens = tokens[ :, len(input_ids[0]):]
    out = tokenizer.batch_decode(tokens, skip_special_tokens=True)        
    return out