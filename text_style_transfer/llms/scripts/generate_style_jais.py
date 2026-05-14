import sys
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "inceptionai/jais-family-13b-chat"
device = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = "CACHE_DIR"

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True,
                                             torch_dtype=torch.bfloat16, cache_dir=cache_dir)

def jais_text_transfer(author_text_example,neutralized_text):
    prompt="The following text is written by a single author: \n"+author_text_example
    prompt+= "\n\nRewrite the following text to make it look like the above author's style. Only provide the rewritten text without explanation or extra text.\n\nRewrite the following:\n"+neutralized_text+"\n\n The rewritten Arabic text is:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=2048,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
        )
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print("response:",flush=True)
    print(response,flush=True)
    return response

def main():
  input_corpus_csv_file=open(sys.argv[1], newline='')
  csvreader = csv.DictReader(input_corpus_csv_file)
  csvfile=open(sys.argv[2], 'w', newline='')
  fieldnames = ['corpus','author_id', 'doc_id','set','content_example','content_neutral','content_styled']
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  writer.writeheader()
  count=0
  for row in csvreader:
    corpus=row['corpus']
    author=row['author_id']
    sampled_doc=row['doc_id']
    dataset=row['set']
    snippet_author_example=row['content_example']
    snippet_neutral=row['content_neutral']
    try:
      snippet_styled=jais_text_transfer(snippet_author_example,snippet_neutral)
      writer.writerow({'corpus': corpus, 'author_id': author, 'doc_id':sampled_doc, 'set':dataset,'content_example':snippet_author_example,'content_neutral':snippet_neutral,'content_styled':snippet_styled})
    except:
      writer.writerow({'corpus': corpus, 'author_id': author, 'doc_id':sampled_doc, 'set':dataset,'content_example':snippet_author_example,'content_neutral':snippet_neutral,'content_styled':"<ERROR>"})
      print("<ERROR>")
    count+=1
    print(str(count)+"/1600", flush=True)

if __name__ == "__main__":
  main()