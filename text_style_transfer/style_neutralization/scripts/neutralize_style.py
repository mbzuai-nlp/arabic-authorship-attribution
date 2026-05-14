import sys
import csv
import openai
openai.api_key = "sk-None-4fZrIegT2J6hc7KNBngsT3BlbkFJyCZf4iT9tBwM1HoNMFpQ"

def gpt_translate(input_text,lang):
  prompt="Translate the following text to "+lang+". Only provide the translation without explanation or extra text.\n\nTranslate the following:\n"
  prompt+=input_text
  print("=========== Translation Prompt =============", flush=True)
  print(prompt, flush=True)
  messages = [
          {'role': 'system', 'content': 'You are an Arabic-English bilingual.'},
          {'role': 'user', 'content': prompt}
      ]
  response = openai.chat.completions.create(
    model = "gpt-4.1",
    temperature = 0.8,
    max_tokens = 2000,
    messages=messages
  )
  prediction=response.choices[0].message.content
  print("=========== Translation Output =============", flush=True)
  print(prediction, flush=True)
  return prediction

def gpt_paraphrase(input_text,lang):
  prompt="Paraphrase the following text in a simple neutral style in "+lang+". No information should be lost in paraphrasing. Only provide the paraphrased text without explanation or extra text.\n\nRewrite the following:\n"
  prompt+=input_text
  print("=========== Paraphrase Prompt =============", flush=True)
  print(prompt, flush=True)
  messages = [
          {'role': 'system', 'content': 'You are a professional writer.'},
          {'role': 'user', 'content': prompt}
      ]
  response = openai.chat.completions.create(
    model = "gpt-4.1",
    temperature = 0.8,
    max_tokens = 2000,
    messages=messages
  )
  prediction=response.choices[0].message.content
  print("=========== Paraphrase Output =============", flush=True)
  print(prediction, flush=True)
  return prediction

def neutralize_text_prompt1(snippet):
  snippet_neutral=gpt_paraphrase(snippet,"Arabic")
  return "-","-",snippet_neutral

def neutralize_text_prompt2(snippet):
  trans_en=gpt_translate(snippet,"English")
  snippet_neutral_en=gpt_paraphrase(trans_en,"English")
  snippet_neutral_ar=gpt_translate(snippet_neutral_en,"Arabic")
  return trans_en,snippet_neutral_en,snippet_neutral_ar

def main():
  input_corpus_csv_file=open(sys.argv[1], newline='')
  csvreader = csv.DictReader(input_corpus_csv_file)
  csvfile=open(sys.argv[2], 'w', newline='')
  fieldnames = ['corpus','author_id', 'doc_id','set','content','translation_en','neutral_en']
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  writer.writeheader()
  neutralize_approach=sys.argv[3] #'rewrite' or 'mt'
  if neutralize_approach=='rewrite':
    print("Neutralizing with rewrite approach")
  else:
    print("Neutralizing with MT approach")
  count=0
  for row in csvreader:
    corpus=row['corpus']
    author=row['author_id']
    sampled_doc=row['doc_id']
    dataset=row['set']
    snippet=row['content']
    if dataset=="test_in" or dataset=="test_out":
      try:
        if neutralize_approach=='rewrite':
          trans_en,snippet_neutral_en,snippet_neutral_ar=neutralize_text_prompt1(snippet)
        else:#mt
          trans_en,snippet_neutral_en,snippet_neutral_ar=neutralize_text_prompt2(snippet)
        dataset_new=dataset+"_neut"
        writer.writerow({'corpus': corpus, 'author_id': author, 'doc_id':sampled_doc, 'set':dataset_new,'content':snippet_neutral_ar,'translation_en':trans_en,'neutral_en':snippet_neutral_en})
      except:
        writer.writerow({'corpus': corpus, 'author_id': author, 'doc_id':sampled_doc, 'set':dataset_new,'content':"<ERROR>",'translation_en':"<ERROR>",'neutral_en':"<ERROR>"})
        print("<ERROR>")
      count+=1
      print(str(count)+"/1600", flush=True)

if __name__ == "__main__":
    main()