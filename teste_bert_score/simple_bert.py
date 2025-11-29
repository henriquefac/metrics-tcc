from transformers import BertTokenizer, BertModel
import torch

MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME, device_map="auto")


def get_embeddings(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Return last hidden states (token-level embeddings)
    return outputs.last_hidden_state


def cossine_similarity(generated_embeddings, reference_embeddings):

    generated_embeddings = torch.nn.functional.normalize(generated_embeddings, dim=-1)
    refe
