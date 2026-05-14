import os
import sys
import csv
import torch
from bert_score import score
from pathlib import Path

input_corpus_csv_file_neutralized=open(sys.argv[1], newline='') #neutralized or styled
csvreader_neutralized = csv.DictReader(input_corpus_csv_file_neutralized)

input_corpus_csv_file_orig=open(sys.argv[2], newline='')
csvreader_orig = csv.DictReader(input_corpus_csv_file_orig)
#header: corpus,author_id,doc_id,snippet_index,set,content

column=sys.argv[3]#"content_styled" for styled or "content" for neutralized

list_neut_txt=[]
list_orig_txt=[]

for row in csvreader_neutralized:
    list_neut_txt.append(row[column])

for row in csvreader_orig:
    if "test" in row['set']:
        list_orig_txt.append(row['content'])

print(len(list_neut_txt))
print(len(list_orig_txt))

_, _, F1_score = score(list_neut_txt, list_orig_txt, lang="others")
print(torch.mean(F1_score))