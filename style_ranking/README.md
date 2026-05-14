# Style Ranking Task
This directory contains scripts and resources for reproducing the style ranking experiments reported in our paper.

Task description: Given a reference text written by a specific author, the task is to rank seven other candidate
text snippets based on their stylistic similarity to the reference. For the reference texts, we sample a
random snippet from each author, while the seven candidates are selected to include a range of author, document, 
and corpus settings. The task involves 80 samples, each consisting of one reference and seven candidate snippets. 
All snippets are sampled from the test set and are truncated to the first 40 words to keep the task manageable for humans.
The data for this task is provided [here](style_ranking_data.csv).

## Human and Automatic Annotations:

- Human Study: 
  - We provide the [human annotations](human_study/style_ranking_human_annotations.tsv) and the [annotation guidelines](human_study/style_ranking_annotation_guidelines.pdf).
- LLMs:
  - To reproduce LLM outputs, run: [llms/scripts/run_llms.sh](llms/scripts/run_llms.sh). The LLM outputs are placed in [llms/output](llms/output)
- AA:
  - To generate the style rankings using the AA model, run: [../authorship_attribution/Contra-X_AR/run_inference_styleRanking.sh](../authorship_attribution/Contra-X_AR/run_inference_styleRanking.sh)

 ## Evaluation:
- Data: We compiled all human and automatic annotations in [style_ranking_all_annotations.tsv](eval/style_ranking_all_annotations.tsv)
- For evaluation, run: [run_eval.sh](eval/run_eval.sh)
