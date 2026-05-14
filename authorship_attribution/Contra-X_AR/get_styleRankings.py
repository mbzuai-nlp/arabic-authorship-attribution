'''
Usage: python get_styleRankings_from_trainedModel.py datasets/llm_ranking_80samples_test.csv styleRanking_humanStudy_embeddings styleRanking_humanStudy_rankings
'''
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import BertClassifier, LogisticRegression

device = "cuda" if torch.cuda.is_available() else "cpu"

def encode_text(text, model, tokenizer):
    encoded = tokenizer(text, max_length=256,padding='max_length', truncation=True, return_tensors='pt')
    x1 = encoded['input_ids'].to(device)
    x2 = encoded['token_type_ids'].to(device)
    x3 = encoded['attention_mask'].to(device)
    with torch.no_grad():
        pred, feats = model((x1, x2, x3), return_feat=True)
    return feats.squeeze(0)
    
def inference_bert(input_csv_file,embeddings_file,model_name,model_dir):

    # initiating architecture
    tokenizer, extractor = None, None
    if 'bert-base' in model_name or 'ALDi' in model_name:
        print("Using BertTokenizer")
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained(model_name)
        extractor = BertModel.from_pretrained(model_name)
    elif 'deberta-base' in model_name:
        print("Using DebertaTokenizer")
        from transformers import DebertaTokenizer, DebertaModel
        tokenizer = DebertaTokenizer.from_pretrained(model_name)
        extractor = DebertaModel.from_pretrained(model_name)
    else:
        print("Using AutoTokenizer")
        from transformers import AutoTokenizer,AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        extractor = AutoModel.from_pretrained(model_name)

    # update extractor
    for param in extractor.parameters():
        param.requires_grad = True

    ngpus, dropout = torch.cuda.device_count(), 0.35
    embed_len=768
    num_authors=40
    num_tokens, hidden_dim, out_dim = 256, 512, num_authors
    model = BertClassifier(extractor, LogisticRegression(embed_len * num_tokens, hidden_dim, out_dim, dropout=dropout))
    model = nn.DataParallel(model).cuda()

    #loading model
    state_dict_loaded = torch.load(model_dir)
    state_dict={}
    for k,v in state_dict_loaded.items():
        state_dict["module."+k]=v
    model.load_state_dict(state_dict)  
    model.eval()

    #encoding texts
    input_corpus_csv_file=open(input_csv_file, newline='') 
    csvreader = csv.DictReader(input_corpus_csv_file)
    snippet_embeddings=[]
    for row in csvreader:
        embedding = encode_text(row['snippet'], model, tokenizer)
        snippet_embeddings.append(embedding)
    torch.save(snippet_embeddings, embeddings_file)
    print("len(snippet_embeddings)",len(snippet_embeddings))
    
def cos_sim(a, b):
    #from sentence_transformers import SentenceTransformer, util
    #cos_sim=util.pytorch_cos_sim(emb1, emb2)
    #using this method to avoid needing two separate environments
    a = a.unsqueeze(0) if a.dim() == 1 else a
    b = b.unsqueeze(0) if b.dim() == 1 else b
    return F.cosine_similarity(a, b)
    
def get_rankings_from_embeddings(embeddings_file,output_rankings_file):
    loaded_tensors=torch.load(embeddings_file, map_location={'cuda:0': 'cpu'})
    index=0
    ranking_overall=[]
    while index<len(loaded_tensors):
        list_cos_similarities=[]
        embedding_main=loaded_tensors[index]
        embedding_1=loaded_tensors[index+1]
        list_cos_similarities.append(cos_sim(embedding_main, embedding_1))
        embedding_2=loaded_tensors[index+2]
        list_cos_similarities.append(cos_sim(embedding_main, embedding_2))
        embedding_3=loaded_tensors[index+3]
        list_cos_similarities.append(cos_sim(embedding_main, embedding_3))
        embedding_4=loaded_tensors[index+4]
        list_cos_similarities.append(cos_sim(embedding_main, embedding_4))
        embedding_5=loaded_tensors[index+5]
        list_cos_similarities.append(cos_sim(embedding_main, embedding_5))
        embedding_6=loaded_tensors[index+6]
        list_cos_similarities.append(cos_sim(embedding_main, embedding_6))
        embedding_7=loaded_tensors[index+7]
        list_cos_similarities.append(cos_sim(embedding_main, embedding_7))
        index+=8
        for e in list_cos_similarities:
            if e<0:
                print("negative")
        ranking=[sorted(list_cos_similarities,reverse=True).index(x)+1 for x in list_cos_similarities]
        #print(list_cos_similarities)
        #print(ranking)
        #print("--------------------------------------")
        ranking_overall.extend(ranking)
    
    output_file=open(output_rankings_file,'w')
    output_file.write("\n".join([str(x) for x in ranking_overall]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", help="Input data file path.")
    parser.add_argument("--output_embeddings", help="File containing manual annotations.")
    parser.add_argument("--output", help="Output file containing the rankings.")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--model_path", help="Model path")

    args = parser.parse_args()
    
    input_csv_file=args.input_data
    embeddings_file=args.output_embeddings #the file where snippet embeddings will be saved
    output_rankings_file=args.output #output file having line separated rankings
    inference_bert(input_csv_file,embeddings_file,args.model,args.model_path)
    get_rankings_from_embeddings(embeddings_file,output_rankings_file)
    
