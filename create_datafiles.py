from datasets import load_dataset
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
import spacy
import torch
import torchtext
from tqdm import tqdm

target_vocab=torch.load('target_vocab.pth')

def target_tok(text):
    return[tok for tok in trivial_tokenize(text,lang='hi')]
    


dataset=load_dataset("cfilt/iitb-english-hindi")


""" source_train_file = open("source_clean_train.txt", "w+", encoding='utf8')
target_train_file = open("target_clean_train.txt", "w+", encoding='utf8')
for translation_pair in tqdm(dataset["train"]["translation"]):
  write=True
  source_sentence = translation_pair["en"]
  target_sentence = translation_pair["hi"]
  target_token=target_tok(target_sentence)
  x=target_vocab.lookup_indices(target_token) 
  if 0 in x:
    write=False

  if write:
    source_train_file.write(source_sentence.strip("\n") + "\n")
    target_train_file.write(target_sentence.strip("\n") + "\n")
source_train_file.close()
target_train_file.close()
 """

source_test_file = open("source_clean_test.txt", "w+", encoding='utf8')
target_test_file = open("target_clean_test.txt", "w+", encoding='utf8')
for translation_pair in tqdm(dataset["test"]["translation"]):
  write=True
  source_sentence = translation_pair["en"]
  target_sentence = translation_pair["hi"]
  target_token=target_tok(target_sentence)
  x=target_vocab.lookup_indices(target_token) 
  if 0 in x:
    write=False

  if write:
    source_test_file.write(source_sentence.strip("\n") + "\n")
    target_test_file.write(target_sentence.strip("\n") + "\n")
source_test_file.close()
target_test_file.close()

""" source_valid_file = open("source_valid.txt", "w+", encoding='utf8')
target_valid_file = open("target_valid.txt", "w+", encoding='utf8')
for translation_pair in dataset["validation"]["translation"]:
  source_sentence = translation_pair["en"]
  target_sentence = translation_pair["hi"]
  source_valid_file.write(source_sentence.strip("\n") + "\n")
  target_valid_file.write(target_sentence.strip("\n") + "\n")
source_valid_file.close()
target_valid_file.close()

source_test_file = open("source_test.txt", "w+", encoding='utf8')
target_test_file = open("target_test.txt", "w+", encoding='utf8')
for translation_pair in dataset["test"]["translation"]:
  source_sentence = translation_pair["en"]
  target_sentence = translation_pair["hi"]
  source_test_file.write(source_sentence.strip("\n") + "\n")
  target_test_file.write(target_sentence.strip("\n") + "\n")
source_test_file.close()
target_test_file.close() """