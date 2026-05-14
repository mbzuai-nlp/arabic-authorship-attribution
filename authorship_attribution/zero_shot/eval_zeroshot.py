import sys
import csv
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import util
#import numpy as np
#import torch.nn.functional as F
#import numpy as np
#import os


def load_model(pretrained_model):
  tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
  model = AutoModel.from_pretrained(pretrained_model,return_dict=True)
  model.eval()
  return model, tokenizer

def get_embeddings_CLS(model,model_inputs):
  #tokenize_input=tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True) #.encode(sentence) #
  with torch.no_grad():
    output=model(**model_inputs, output_hidden_states=True) # ,labels=input_ids, output_hidden_states=True
    last_hidden_states = output.last_hidden_state
    '''
    pooler_output: Last layer hidden-state of the first token of the sequence (classification token) after further processing 
    '''
    cls_representation = last_hidden_states[:,0,:]
    return cls_representation

'''
1- create author-snippets dict
dict={author:{train:[], tune_in:[],dev_in:[],test_in:[],tune_out:[],dev_out:[],test_out:[]}}
2- for each author, get average embedding of all embeddings
3- for each eval snippet, get closest authors
'''

def get_results_for_eval_set(author_set_snippets_dict,author_train_aggregated_embedding,eval_set):
  count_accuracyAt1=0
  count_accuracyAt5=0
  L_ranking=[]
  print("============> EVAL: "+eval_set)
  for ground_truth_author in author_train_aggregated_embedding:
    eval_snippets_embeddings=author_set_snippets_dict[ground_truth_author][eval_set]
    for eval_snippet_embedding in eval_snippets_embeddings:
      map_sim_to_all_authors={}
      #using cosine similarity to get the most similar author embedding
      for author_comparison in author_train_aggregated_embedding:
        sim=util.pytorch_cos_sim(eval_snippet_embedding, author_train_aggregated_embedding[author_comparison])
        map_sim_to_all_authors[author_comparison]=sim
      map_sim_to_all_authors_sorted=sorted(map_sim_to_all_authors,key=map_sim_to_all_authors.get,reverse=True)
      if ground_truth_author in map_sim_to_all_authors_sorted:
        ranking=map_sim_to_all_authors_sorted.index(ground_truth_author)+1
        L_ranking.append(ranking)
        if ranking==1: count_accuracyAt1+=1
        if ranking<=5: count_accuracyAt5+=1
      else:
        print("XXXXXXXXXXXX")
        print(ground_truth_author)
  mrr=sum(L_ranking)/len(L_ranking)
  return count_accuracyAt1/len(L_ranking),count_accuracyAt5/len(L_ranking),mrr

def get_embeddings(input_corpus_csv_file,pretrained_model):
  input_corpus_csvreader = csv.DictReader(input_corpus_csv_file)
  model, tokenizer=load_model(pretrained_model)
  print("======== Model: "+pretrained_model)
  all_snippets_embeddings=[]
  index=0
  for row in input_corpus_csvreader:
    snippet=row['content']
    model_inputs = tokenizer(snippet, return_tensors='pt', max_length=512, truncation=True)
    snippet_embedding=get_embeddings_CLS(model,model_inputs).squeeze()
    all_snippets_embeddings.append(snippet_embedding)
    index+=1
    print("Progress: "+str(index)+"/12000")
  return all_snippets_embeddings

def main():
  input_corpus_csv_file=open(sys.argv[1], newline='')
  
  pretrained_model=sys.argv[2]

  all_snippets_embeddings=get_embeddings(input_corpus_csv_file,pretrained_model)
  #print("num_embeddings: "+str(len(all_snippets_embeddings)))

  #1- create author-snippets dict
  input_corpus_csv_file=open(sys.argv[1], newline='')
  input_corpus_csvreader = csv.DictReader(input_corpus_csv_file)
  author_set_snippets_dict={}
  index=0
  for row in input_corpus_csvreader:
    author=row['author_id']
    dataset=row['set']
    embedding=all_snippets_embeddings[index]
    if author not in author_set_snippets_dict:
      author_set_snippets_dict[author]={"train":[],"tune_in":[],"dev_in":[],"test_in":[],"tune_out":[],"dev_out":[],"test_out":[]}
    author_set_snippets_dict[author][dataset].append(embedding)
    index+=1

  #2- for each author, get average embedding of all embeddings
  author_train_aggregated_embedding={}
  for author in author_set_snippets_dict:
    train_snippet_embeddings=author_set_snippets_dict[author]["train"]
    author_train_aggregated_embedding[author]=torch.stack(train_snippet_embeddings).mean(dim=0)

  #3- for each eval snippet, get closest authors
  print("test_in",get_results_for_eval_set(author_set_snippets_dict,author_train_aggregated_embedding,"test_in"))
  print("test_out",get_results_for_eval_set(author_set_snippets_dict,author_train_aggregated_embedding,"test_out"))

if __name__=="__main__":
  main()
