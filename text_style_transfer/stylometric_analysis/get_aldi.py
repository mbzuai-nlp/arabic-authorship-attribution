import sys
import csv
from camel_tools.utils.charsets import AR_LETTERS_CHARSET
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "AMR-KELEG/Sentence-ALDi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def compute_aldi_score(sentence):
  inputs = tokenizer(sentence, return_tensors="pt")
  outputs = model(**inputs)
  logits = outputs.logits
  return min(max(0, logits[0][0].item()), 1)

def stringContainsArabic(word):
    for c in word:
        if c in AR_LETTERS_CHARSET:
            return True
    return False

def get_aldi_scores(csvreader,writer):
  snippets_aldi_levels=[]
  content_case=sys.argv[3] #content_styled/content_example
  count=0
  count_undefined=0
  for row in csvreader:
    if content_case!="content" or (content_case=="content" and 'test' in row['set']):
      content=row[content_case]
      sentences_aldi_levels=[]
      snippet_aldi_level=0
      for sentence in content.split("\n"):
        if len(sentence.strip())>0 and stringContainsArabic(sentence):
          try: #trying the whole snippet
            sentence_aldi_level=compute_aldi_score(sentence)
          except:
            sentence_words=sentence.split(" ")
            word_index=0
            chunks_aldi=[]
            try: #trying with chunks of 200 words
              while word_index<len(sentence_words):
                chunk=" ".join(sentence_words[word_index:min(word_index+200,len(sentence_words))])
                chunks_aldi.append(compute_aldi_score(chunk))
                word_index+=200
            except:
                chunks_aldi=[]
                while word_index<len(sentence_words):
                  try: #trying with chunks of 100 words
                    chunk=" ".join(sentence_words[word_index:min(word_index+100,len(sentence_words))])
                    chunks_aldi.append(compute_aldi_score(chunk))
                  except:
                    chunk=" ".join(sentence_words[word_index:min(word_index+100,len(sentence_words))])
                    try: chunks_aldi.append(compute_aldi_score(chunk[:512]))
                    except: chunks_aldi.append(compute_aldi_score(chunk[:510])) #around 2 cases to be handled
                  word_index+=100
            sentence_aldi_level=sum(chunks_aldi)/len(chunks_aldi)
          sentences_aldi_levels.append(sentence_aldi_level)
      if len(sentences_aldi_levels)>0:
        snippet_aldi_level=sum(sentences_aldi_levels)/len(sentences_aldi_levels)
        snippets_aldi_levels.append(snippet_aldi_level)
      else:
        count_undefined+=1
      writer.writerow({'snippet_id': count+1, 'ALDi': snippet_aldi_level})
      count+=1
    print("Progress: "+str(count)+"/1600")
  print("count",count)
  print("count_undefined",count_undefined)

def main():
  input_corpus_csv_file=open(sys.argv[1], newline='')
  csvreader = csv.DictReader(input_corpus_csv_file)
  csvfile=open(sys.argv[2], 'w', newline='')
  fieldnames = ['snippet_id','ALDi']
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  writer.writeheader()
  get_aldi_scores(csvreader,writer)

if __name__=="__main__":
  main()