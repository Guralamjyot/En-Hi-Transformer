from datasets import load_dataset
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
import torch
import re
from tqdm import tqdm

target_vocab=torch.load('target_vocab.pth')

def target_tok(text):
    return[tok for tok in trivial_tokenize(text,lang='hi')]
    

re_for_t='[a-zA-Z]'
dataset=load_dataset("cfilt/iitb-english-hindi")


source_train_small = open("source_train_small.txt", "w+", encoding='utf8')
source_train_mid = open("source_train_mid.txt", "w+", encoding='utf8')
source_train_large = open("source_train_large.txt", "w+", encoding='utf8')
source_train_xl = open("source_train_xl.txt", "w+", encoding='utf8')
target_train_small = open("target_train_small.txt", "w+", encoding='utf8')
target_train_mid = open("target_train_mid.txt", "w+", encoding='utf8')
target_train_large = open("target_train_large.txt", "w+", encoding='utf8')
target_train_xl = open("target_train_xl.txt", "w+", encoding='utf8')

for translation_pair in tqdm(dataset["train"]["translation"]):
  source_sentence = translation_pair["en"]
  target_sentence = translation_pair["hi"]
  if(re.search(re_for_t,target_sentence)):
     continue

  if len(target_sentence)<21 and len(target_sentence)>0 and len(source_sentence)<21:
    source_train_small.write(source_sentence.strip("\n") + "\n")
    target_train_small.write(target_sentence.strip("\n") + "\n")
  elif len(target_sentence)>20 and len(target_sentence)<41 and len(source_sentence)<41:
    source_train_mid.write(source_sentence.strip("\n") + "\n")
    target_train_mid.write(target_sentence.strip("\n") + "\n")
  elif len(target_sentence)>40 and len(target_sentence)<81 and len(source_sentence)<81:
     source_train_large.write(source_sentence.strip("\n") + "\n")
     target_train_large.write(target_sentence.strip("\n") + "\n")
  elif len(target_sentence)>80 and len(target_sentence)<161 and len(source_sentence)<161:
     source_train_xl.write(source_sentence.strip("\n") + "\n")
     target_train_xl.write(target_sentence.strip("\n") + "\n")
     
     
source_train_small.close()
source_train_mid.close()
source_train_large.close()
source_train_xl.close()
target_train_small.close()
target_train_mid.close()
target_train_large.close()
target_train_xl.close()



source_valid_file = open("source_valid.txt", "w+", encoding='utf8')
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
target_test_file.close()