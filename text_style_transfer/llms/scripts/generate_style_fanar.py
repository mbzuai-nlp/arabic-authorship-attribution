import sys
import csv

from transformers import AutoModelForCausalLM, AutoTokenizer
cache_dir = "CACHE_DIR"

allam_model = AutoModelForCausalLM.from_pretrained("QCRI/Fanar-1-9B-Instruct",cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained("QCRI/Fanar-1-9B-Instruct",use_fast=False) 
allam_model = allam_model.to('cuda')

def text_transfer(author_text_example,neutralized_text):
    prompt="The following text is written by a single author: \n"+author_text_example
    prompt+= "\n\nRewrite the following text to make it look like the above author's style. Only provide the rewritten text without explanation or extra text.\n\nRewrite the following:\n"+neutralized_text
    messages = [
        {"role": "system", "content": "You are a professional writer. "},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(inputs, return_tensors='pt', return_token_type_ids=False)
    inputs = {k: v.to('cuda') for k,v in inputs.items()}
    response = allam_model.generate(**inputs, max_new_tokens=256)
    output=tokenizer.decode(response[0], skip_special_tokens=True)
    print(output,flush=True)
    return output

def main():
  input_corpus_csv_file=open(sys.argv[1], newline='')
  csvreader = csv.DictReader(input_corpus_csv_file)
  csvfile=open(sys.argv[2], 'w', newline='')
  fieldnames = ['corpus','author_id', 'doc_id','set','content_example','content_neutral','content_styled']
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  writer.writeheader()
  count=0
  print("Start prompting")
  for row in csvreader:
    corpus=row['corpus']
    author=row['author_id']
    sampled_doc=row['doc_id']
    dataset=row['set']
    snippet_author_example=row['content_example']
    snippet_neutral=row['content_neutral']
    try:
      snippet_styled=text_transfer(snippet_author_example,snippet_neutral)
      writer.writerow({'corpus': corpus, 'author_id': author, 'doc_id':sampled_doc, 'set':dataset,'content_example':snippet_author_example,'content_neutral':snippet_neutral,'content_styled':snippet_styled})
    except:
      writer.writerow({'corpus': corpus, 'author_id': author, 'doc_id':sampled_doc, 'set':dataset,'content_example':snippet_author_example,'content_neutral':snippet_neutral,'content_styled':"<ERROR>"})
      print("<ERROR>")
    count+=1
    print(str(count)+"/1600", flush=True)

if __name__ == "__main__":
  main()