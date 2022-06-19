This tutorial presents how to train and evaluate a model using our code for the 1st task of DeftEval competition.

Firstly, download the Deft corpus from [here](https://github.com/adobe-research/deft_corpus) and install the project [requirements](https://github.com/avramandrei/UPB-at-SemEval-2020-Task-6-Pretrained-Language-Models-for-DefinitionExtraction/blob/master/requirements.txt).

Then start training using the `train.py` script.

```
python3 train.py [-h] [--fine_tune FINE_TUNE] [--hidden_size HIDDEN_SIZE] [--device DEVICE] [--batch_size BATCH_SIZE] 
                 lang_model train_data dev_data save_path
```

For testing the fine-tuned model first download the trained model from [here](https://drive.google.com/file/d/1EkIqMW3vdSYTZ_pDNAp9p422xleLPGck/view?usp=sharing) and then place it in the `model/` directory.

Then predict any sentence with following script:

```
 python predict.py [-h] sentence
```
