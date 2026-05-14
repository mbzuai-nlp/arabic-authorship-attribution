import sys
import csv
import random
random.seed(20)

def main():
  dict_authors_documents={}
  dict_documents_train_snippets={}

  input_corpus_csv_file=open(sys.argv[1], newline='')
  csvreader = csv.DictReader(input_corpus_csv_file)

  for row in csvreader:
    corpus=row['corpus']
    author=row['author_id']
    sampled_doc=row['doc_id']
    dataset=row['set']
    snippet=row['content']
    if dataset=="train":
      if author not in dict_authors_documents:
        dict_authors_documents[author]=set()
      dict_authors_documents[author].add(sampled_doc)
      
      if sampled_doc not in dict_documents_train_snippets:
        dict_documents_train_snippets[sampled_doc]=[]
      dict_documents_train_snippets[sampled_doc].append(snippet)

  input_corpus_csv_file_neutral=open(sys.argv[2], newline='')
  csvreader_neutral = csv.DictReader(input_corpus_csv_file_neutral)

  csvfile=open(sys.argv[3], 'w', newline='')
  fieldnames = ['corpus','author_id', 'doc_id','set','content_example','content_neutral']
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  writer.writeheader()

  for row in csvreader_neutral:
    corpus=row['corpus']
    author=row['author_id']
    sampled_doc=row['doc_id']
    snippet=row['content']
    dataset=row['set']
    random_train_snippet=""
    if dataset=="test_in_neut":
      random_train_snippet=random.choice(dict_documents_train_snippets[sampled_doc])
    elif dataset=="test_out_neut":
      random_author_doc=random.choice(sorted(list(dict_authors_documents[author])))
      random_train_snippet=random.choice(dict_documents_train_snippets[random_author_doc])
    else:
      print("XXXXXXXXXXX")
    writer.writerow({'corpus': corpus, 'author_id': author, 'doc_id':sampled_doc, 'set':dataset,'content_example':random_train_snippet,'content_neutral':snippet})
  
if __name__ == "__main__":
    main()