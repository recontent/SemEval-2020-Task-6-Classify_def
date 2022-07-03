from transformers import AutoModelForTokenClassification, AutoTokenizer


def find_definition_term(sentence, label_list, model_directory, if_print=True):
    model = AutoModelForTokenClassification.from_pretrained(model_directory)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)

    encoding = tokenizer([sentence], return_tensors='pt')
    outputs = model(**encoding)
    logits = outputs.logits
    predicted_label_classes = logits.argmax(-1)
    predicted_labels = [label_list[id] for id in predicted_label_classes.squeeze().tolist()]

    word_level = []
    for word_id in encoding.words():
        if word_id is not None:
            start, end = encoding.word_to_tokens(word_id)
            word_level.append((start, end))

    outputs = []

    for id in encoding.input_ids.squeeze().tolist():
        outputs.append(tokenizer.decode([id]).strip())

    perv_word = ""
    output = []

    for tup in word_level:
        word = "".join(outputs[tup[0]:tup[-1]]).ljust(12)
        if word != perv_word:
            prediction = predicted_labels[tup[0]]
            prediction = "Term" if "Term" in prediction else "Definition" if "Definition" in prediction else "O"
            if if_print:
                print(word, f" ---------> {prediction}")
            output.append((word, prediction))
            perv_word = word

    return output


if __name__ == "__main__":
    labels = ['B-Term', 'I-Term', 'B-Definition', 'I-Definition', 'B-Alias-Term', 'I-Alias-Term',
              'B-Referential-Definition', 'I-Referential-Definition', 'B-Referential-Term', 'I-Referential-Term',
              'B-Qualifier', 'I-Qualifier', 'O']

    model_path = "./model/"

    input_sent = "I'm really interested in dota 2. it is a strategy base game which people compete in two teams."

    output = find_definition_term(input_sent, labels, model_path)
