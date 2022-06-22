from transformers import AutoModelForTokenClassification, AutoTokenizer

labels = ['B-Term', 'I-Term', 'B-Definition', 'I-Definition', 'B-Alias-Term', 'I-Alias-Term',
          'B-Referential-Definition', 'I-Referential-Definition', 'B-Referential-Term', 'I-Referential-Term',
          'B-Qualifier', 'I-Qualifier', 'O']

model_path = "./model"

model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

sentence = "The workflow of GRU is same as RNN but the difference is in the operations inside the GRU unit. Lets see " \
           "the architecture of it."

encoding = tokenizer([sentence], return_tensors='pt')
outputs = model(**encoding)
logits = outputs.logits
predicted_label_classes = logits.argmax(-1)
predicted_labels = [labels[id] for id in predicted_label_classes.squeeze().tolist()]

for id, label in zip(encoding.input_ids.squeeze().tolist(), predicted_labels):
    print("%s ------->  %s" % (tokenizer.decode([id]).strip().ljust(12), label))
