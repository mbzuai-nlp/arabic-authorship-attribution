import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score



def calc_kappa(a1,a2,corpus_filtered,df):
  rankings_annotator1 = df[a1]
  rankings_annotator2 = df[a2]
  rankings_annotator1_filtered=[]
  rankings_annotator2_filtered=[]
  sampling_cases=df['sampling_case']
  authors=df['author']
  count_main=0
  for index in range(len(rankings_annotator1)):
    sampling_case=sampling_cases[index]
    if sampling_case=='main':
      corpus=authors[index].split("_")[0]
      count_main+=1
    else:
      if corpus_filtered=='all' or corpus==corpus_filtered:
        rankings_annotator1_filtered.append(rankings_annotator1[index])
        rankings_annotator2_filtered.append(rankings_annotator2[index])
  # Calculate weighted kappa
  #weighted_kappa_none = cohen_kappa_score(annotator1_filtered, annotator2_filtered)
  #weighted_kappa_linear = cohen_kappa_score(annotator1_filtered, annotator2_filtered, weights='linear')
  weighted_kappa_quadratic = cohen_kappa_score(rankings_annotator1_filtered,rankings_annotator2_filtered, weights='quadratic')
  return weighted_kappa_quadratic

def main():
  df = pd.read_csv(sys.argv[1], sep='\t')

  #calculate quadratic weighted kappa between both human annotators
  weighted_kappa_quadratic_A1=calc_kappa('Human_Annotator1','Human_Annotator2','all',df)
  print("Quadratic weighted kappa between human annotators:")
  print(f"{weighted_kappa_quadratic_A1:.2f}")

  #calculate quadratic weighted kappa between AA_Model/LLMs and human annotators
  print("Quadratic weighted kappa between AA_Model/LLM outputs and human annotators:")
  for model in ['AA_model','GPT','Deepseek','Gemini','Llama','Fanar','Jais']:
    print("LLM: ",model)
    for corpus in ['all','Hindawi','Gumar']:
      weighted_kappa_quadratic_A1=calc_kappa('Human_Annotator1',model,corpus,df)
      weighted_kappa_quadratic_A2=calc_kappa('Human_Annotator2',model,corpus,df)
      weighted_kappa_quadratic_avg=(weighted_kappa_quadratic_A1+weighted_kappa_quadratic_A2)/2
      print("corpus: ",corpus)
      print(f"{weighted_kappa_quadratic_avg:.2f}")
    print("------------------------------")
  

if __name__ == "__main__":
    main()

