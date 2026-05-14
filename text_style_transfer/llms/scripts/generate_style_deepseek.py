import sys
import csv
import requests
import json

# Replace 'YOUR_API_KEY' with your actual DeepSeek API key
API_KEY = 'YOUR_API_KEY'
API_URL = 'https://api.deepseek.com/chat/completions'

HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}
def deepseek_text_transfer(author_text_example,neutralized_text):
  prompt="The following text is written by a single author: \n"+author_text_example
  prompt+= "\n\nRewrite the following text to make it look like the above author's style. Only provide the rewritten text without explanation or extra text.\n\nRewrite the following:\n"+neutralized_text
  data = {
      'model': 'deepseek-chat',
      'messages': [
          {'role': 'system', 'content': 'You are a professional writer.'},
          {'role': 'user', 'content': prompt}
      ],
      'stream': False  # Set to True for streaming responses
  }
  response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
  if response.status_code == 200:
      result = response.json()
      return result['choices'][0]['message']['content'].strip()
  else:
      raise Exception(f"Error {response.status_code}: {response.text}")

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
        snippet_styled=deepseek_text_transfer(snippet_author_example,snippet_neutral)
        writer.writerow({'corpus': corpus, 'author_id': author, 'doc_id':sampled_doc, 'set':dataset,'content_example':snippet_author_example,'content_neutral':snippet_neutral,'content_styled':snippet_styled})
        count+=1
        print(str(count)+"/1600", flush=True)

if __name__ == "__main__":
    main()