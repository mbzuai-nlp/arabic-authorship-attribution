import sys
import csv

def main():
  input_csv_file=open(sys.argv[1], newline='')
  csvreader = csv.DictReader(input_csv_file)
  output_csvfile=open(sys.argv[1].replace(".csv","_processed.csv"), 'w', newline='')
  fieldnames = ['corpus','author_id','doc_id','set','content_example','content_neutral','content_styled']
  writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
  writer.writeheader()

  count=0
  count_error=0
  count_empty=0
  for row in csvreader:
    corpus=row['corpus']
    author=row['author_id']
    sampled_doc=row['doc_id']
    dataset=row['set']
    content_example=row['content_example']
    content_neutral=row['content_neutral']
    content_styled=row['content_styled']
    if "jais" in sys.argv[1].lower():
      #response=content_styled.split(content_neutral) #response contains the prompt
      response=content_styled.split("The rewritten Arabic text is:") #response contains the prompt
      if len(response)>1:
          content_styled=response[1].strip()
      else:
          print("empty")
          print(content_styled)
          print("--------------------------")
          print(content_neutral)
          print("==========================")
          count_empty+=1
          content_styled=content_neutral
          count_error+=1
    elif "fanar" in sys.argv[1].lower():
        content_styled=content_styled.split("<start_of_turn>model")[1]
        if len(content_styled.strip())==0:
          content_styled=content_neutral
          count_error+=1
    if "<ERROR>" in content_styled:
          content_styled=content_neutral
          count_error+=1
    writer.writerow({'corpus': corpus, 'author_id': author, 'doc_id':sampled_doc, 'set':dataset,
                    'content_example':content_example,'content_neutral':content_neutral,'content_styled':content_styled})
    count+=1
  print(count)
  print("count_error",count_error)
  print("count_empty",count_empty)

if __name__ == "__main__":
  main()