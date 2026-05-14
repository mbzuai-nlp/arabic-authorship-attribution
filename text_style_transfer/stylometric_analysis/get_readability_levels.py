import sys
import csv
from transformers import pipeline
from camel_tools.utils.charsets import AR_LETTERS_CHARSET

def stringContainsArabic(word):
    for c in word:
        if c in AR_LETTERS_CHARSET:
            return True
    return False

def get_readability_levels(csvreader,writer):
  readability = pipeline("text-classification", model="CAMeL-Lab/readability-arabertv02-word-CE")
  snippets_readability_levels=[]
  content_case=sys.argv[3] #column header; content_styled/content_example
  count=0
  count_undefined=0
  for row in csvreader:
    if content_case!="content" or (content_case=="content" and 'test' in row['set']): #styled snippet or original/neutralized test snippet
      content=row[content_case]
      sentences_readability_levels=[]
      for sentence in content.split("\n"):
        if len(sentence.strip())>0 and stringContainsArabic(sentence):
          try: #trying the whole snippet
            sentence_readability_level=int(readability(sentence)[0]['label'][6:])+1
          except:
            sentence_words=sentence.split(" ")
            word_index=0
            chunks_readability_levels=[]
            try: #trying with chunks of 200 words
              while word_index<len(sentence_words):
                chunk=" ".join(sentence_words[word_index:min(word_index+200,len(sentence_words))])
                chunks_readability_levels.append(int(readability(chunk)[0]['label'][6:])+1)
                word_index+=200
            except: 
                chunks_readability_levels=[]
                while word_index<len(sentence_words):
                  try: #trying with chunks of 100 words
                    chunk=" ".join(sentence_words[word_index:min(word_index+100,len(sentence_words))])
                    chunks_readability_levels.append(int(readability(chunk)[0]['label'][6:])+1)
                  except:
                    chunk=" ".join(sentence_words[word_index:min(word_index+100,len(sentence_words))])
                    try: chunks_readability_levels.append(int(readability(chunk[:512])[0]['label'][6:])+1)
                    except: chunks_readability_levels.append(int(readability(chunk[:510])[0]['label'][6:])+1)  #around 2 cases to be handled
                  word_index+=100
            sentence_readability_level=sum(chunks_readability_levels)/len(chunks_readability_levels)
          sentences_readability_levels.append(sentence_readability_level)
      if len(sentences_readability_levels)>0: snippet_readability_level=sum(sentences_readability_levels)/len(sentences_readability_levels)
      else: 
        print(content)
        print("----------------------------------")
        snippet_readability_level=0
        count_undefined+=1
      writer.writerow({'snippet_id': count+1, 'readability_level': snippet_readability_level})
      snippets_readability_levels.append(snippet_readability_level)
    count+=1
    print("Progress: "+str(count)+"/1600")
  print("count",count)
  print("count_undefined",count_undefined)

def main():
  input_corpus_csv_file=open(sys.argv[1], newline='')
  csvreader = csv.DictReader(input_corpus_csv_file)
  csvfile=open(sys.argv[2], 'w', newline='')
  fieldnames = ['snippet_id','readability_level']
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  writer.writeheader()
  get_readability_levels(csvreader,writer)

if __name__=="__main__":
  main()