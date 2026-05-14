import sys
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "QCRI/Fanar-1-9B-Instruct"
cache_dir = "CACHE_DIR"
tokenizer = AutoTokenizer.from_pretrained("QCRI/Fanar-1-9B-Instruct")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

def is_valid_ranking_output(ranking):
    list_rankings=ranking.split(",")
    list_rankings_sorted=sorted(list_rankings)
    if len(list_rankings_sorted)!=7:
        return False
    for i in range(1,8):
        if int(list_rankings_sorted[i-1])!=i:
            return False
    return True
    
def execute_prompt(list_text_snippets):
  prompt="You are an Arabic professional writer. You are given a reference text written by a specific author. Your task is to rank 7 other text snippets based on their stylistic similarity to the reference text. The most stylistically similar snippet should receive a ranking of 1, and the least similar should receive a ranking of 7.\n"
  prompt+="General Instructions:\n"
  prompt+="Focus on style, not on topic or content.\n"
  prompt+="Use the provided stylometric features to guide your judgment.\n"
  prompt+="Assign each rank only once (i.e., no two snippets should have the same rank).\n"
  prompt+="Be consistent and thorough in your comparison.\n"
  prompt+="If unsure, consider which snippet could have been written by the same author as the reference.\n"
  prompt+="Stylometric Features to Consider:\n"
  prompt+="Surface Features: examine low-level characteristics of the text such as the average sentence length, punctuation usage, repetition of letters, as well usage of diacritics, elongations, and hamzas. For diacritics, assess their usage, distinguishing between syntactic and lexical diacritics.\n"
  prompt+="Lexical Features: look at word choices and how varied they are, checking word frequencies distribution, vocabulary richness and complexity, and orthographic spelling of words.\n"
  prompt+="Syntactic Features: assess the grammatical structure and complexity of text, including morphosyntactic features, part-of-speech distribution, sentence structure complexity, tense and narrative voice (first, second, or third person).\n"
  prompt+="Level of Dialectness: evaluate the level of dialectness, ranging from formal Modern Standard Arabic to high degree of informal colloquialism.\n"
  prompt+="Stylistic and Aesthetic Features: notice expressive elements and literary style such as presence of poetic tone as well usage of vivid imagery, metaphors, sarcasm and humor.\n"
  prompt+="Provided the following reference text and 7 other text snippets, return the ranking of the 7 snippets based on their stylistic similarity to the reference text. The output should only contain comma-separated rankings in the same order of the text snippets.\n"
  prompt+="Reference text:\n"
  prompt+=list_text_snippets[0]+"\n"
  prompt+="Text Snippet 1:\n"
  prompt+=list_text_snippets[1]+"\n"
  prompt+="Text Snippet 2:\n"
  prompt+=list_text_snippets[2]+"\n"
  prompt+="Text Snippet 3:\n"
  prompt+=list_text_snippets[3]+"\n"
  prompt+="Text Snippet 4:\n"
  prompt+=list_text_snippets[4]+"\n"
  prompt+="Text Snippet 5:\n"
  prompt+=list_text_snippets[5]+"\n"
  prompt+="Text Snippet 6:\n"
  prompt+=list_text_snippets[6]+"\n"
  prompt+="Text Snippet 7:\n"
  prompt+=list_text_snippets[7]+"\n"
  prompt+= "Now provide the comma-separated rankings without explanation or extra text."
  messages = [{"role": "user", "content": prompt},]
  inputs = tokenizer.apply_chat_template(messages, tokenize=False, return_tensors="pt")
  outputs = model.generate(**tokenizer(inputs, return_tensors="pt", return_token_type_ids=False), max_new_tokens=256)
  output=tokenizer.decode(outputs[0], skip_special_tokens=True)
  return output


def main():
    input_corpus_csv_file=open(sys.argv[1], newline='')
    csvreader = csv.DictReader(input_corpus_csv_file)
    output_file=open(sys.argv[2], 'w') 
    count=0
    count_incorrect_output_template=0
    snippet_samples=[]
    for row in csvreader:
        snippet=row['snippet']
        snippet_samples.append(snippet)
        count+=1
        successful_generation=False
        generation_trial_count=0
        if count%8==0:
            print("case:"+str(int(count/8)))
            while not successful_generation:
                prompt_output=execute_prompt(snippet_samples)
                prompt_output=prompt_output.split("\n")[-1]#extract answer from the last line
                prompt_output=prompt_output.replace(" ","")
                print(prompt_output)
                if is_valid_ranking_output(prompt_output):
                    successful_generation=True
                    output_file.write("\n".join(prompt_output.split(","))+"\n")
                else:
                    print("========> retrying")
                generation_trial_count+=1
                if generation_trial_count==20 and not successful_generation:
                    output_file.write("<ERROR>\n")
                    count_incorrect_output_template+=1
                    break
            snippet_samples=[]
    print("count_incorrect_output_template",count_incorrect_output_template)

if __name__ == "__main__":
    main()
    