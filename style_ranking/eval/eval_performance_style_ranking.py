import sys
import pandas as pd
#from sklearn.metrics import cohen_kappa_score
#from scipy.stats import spearmanr

def calc_performance(source,df):
    rankings_hindawi=[]
    rankings_gumar=[]
    same_author_aa_percentage=[]
    same_author_aa_percentage_hindawi=[]
    same_author_aa_percentage_gumar=[]
    rankings=df[source]
    sampling_cases=df['sampling_case']
    authors=df['author']
    count_main=0
    for index in range(len(rankings)):
        sampling_case=sampling_cases[index]
        if sampling_case=='main':
            count_main+=1
            corpus=authors[index].split("_")[0]
        else:
            ranking=int(rankings[index])
            if corpus=='Hindawi':
                rankings_hindawi.append(ranking)
            elif corpus=='Gumar':
                rankings_gumar.append(ranking)
            if sampling_case.startswith("snippet_same_author"):
                if ranking>0 and ranking<=3: 
                    same_author_aa_percentage.append(1)
                else:
                    same_author_aa_percentage.append(0)
                if corpus=='Hindawi':
                    if ranking>0 and ranking<=3: same_author_aa_percentage_hindawi.append(1)
                    else: same_author_aa_percentage_hindawi.append(0)
                elif corpus=='Gumar':
                    if ranking>0 and ranking<=3: same_author_aa_percentage_gumar.append(1)
                    else: same_author_aa_percentage_gumar.append(0)
    return same_author_aa_percentage,same_author_aa_percentage_hindawi,same_author_aa_percentage_gumar

def main():
    df = pd.read_csv(sys.argv[1], sep='\t')

    #calculating percentage of snippets belonging to same author assigned ranking between 1-3 (%Same Author in Top-3)
    for source in ['Human_Annotator1','Human_Annotator2','AA_model','Deepseek','GPT','Llama','Gemini','Fanar','Jais']:
        print(source+":")
        same_author_aa_percentage,same_author_aa_percentage_hindawi,same_author_aa_percentage_gumar=calc_performance(source,df)
        print(f"percentage_same_author_top3_all: {100*sum(same_author_aa_percentage)/len(same_author_aa_percentage):.1f}%")
        print(f"percentage_same_author_top3_hindawi {100*sum(same_author_aa_percentage_hindawi)/len(same_author_aa_percentage_hindawi):.1f}%")
        print(f"percentage_same_author_top3_gumar {100*sum(same_author_aa_percentage_gumar)/len(same_author_aa_percentage_gumar):.1f}%")
        print("--------------------------------------------------------------------------")

if __name__ == "__main__":
    main()