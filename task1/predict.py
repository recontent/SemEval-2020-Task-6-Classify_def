from torch.nn.utils.rnn import pad_sequence
from model import LangModelWithDense
from utils import parse_lang_model
from process import clean_data
import pandas as pd
import argparse
import torch


def process_test_sentences(df, tokenizer, fine_tune, device):
    """
    Cleaning and padding test sentences.

    Arguments:
        df: dataframe contain sentence as column 0.
        tokenizer: model tokenizer.
        fine_tune: fine-tuned or frozen
        device: cuda or cpu

    Returns:
        x: an array of sentence token ids.
        mask: array of mask sentence to specify paddings.
    """
    mask = []

    df = clean_data(df, tokenizer, fine_tune)[0]

    x = df.values

    tokens = []
    for i in range(x.shape[0]):
        t = torch.tensor(tokenizer.encode(x[i], add_special_tokens=True))
        tokens.append(t)
        mask.append(torch.ones_like(t))

    x = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    mask = pad_sequence(mask, batch_first=True, padding_value=0).to(device)

    return x, mask


def predict(sent, model_name="roberta-base", model_path="./model/task1-fine-tuned.pth", hidden_size=512, device='cuda', fine_tune=True):
    """
    Predict the label of the sentences.

    Arguments:
        sent: string, input sentence.
        model_name: string, name of the model.
        model_path: string, path to the fine-tuned model.
        fine_tune: bool, fine-tune or freeze.
        device: string, gpu or cpu
        hidden_size: int, hidden size if model.

    Return:
        output: int, 0 or 1 as label of the sentence.
    """

    lang_model, tokenizer, lm_emb_size = parse_lang_model(model_name)
    vocab_size = len(tokenizer)

    model = LangModelWithDense(lang_model, vocab_size, lm_emb_size, hidden_size=hidden_size, fine_tune=fine_tune).to(device)
    model.load_state_dict(torch.load(model_path))

    data_list = [[sent, None]]
    test_df = pd.DataFrame(data_list)

    tokens, masks = process_test_sentences(test_df, tokenizer, fine_tune=True, device=device)

    output = model.forward(tokens, masks)
    output = torch.tensor([0 if x < 0.5 else 1 for x in output])

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence")
    args = parser.parse_args()

    prediction = predict(args.sentence)
    print(int(prediction))
