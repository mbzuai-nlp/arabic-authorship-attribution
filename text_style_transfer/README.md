# Text Style Transfer
This directory contains scripts and resources for reproducing the text style transfer experiments reported in our paper.

## Style Neutralization:
- To perform style neutralization to the original snippet, run the scripts in: [style_neutralization/scripts/neutralize_style.sh](style_neutralization/scripts/neutralize_style.sh). The neutralized output is placed under [style_neutralization/data](style_neutralization/data).

## TST LLM Generations:
- Data: To prepare data needed for this task, run: [prepare_data.sh](llms/scripts/prepare_data.sh). 
  The prepared data is provided [here](llms/data/dataset_A3Dtest_neutralized_mt_wTrainExamples.csv).
  This contains the test snippets, where for each snippet, we provide:
  (1) the style neutralized snippet obtained through the translation pipeline.
  (2) the author reference snippet obtained from the train set.
- To reproduce LLM outputs, run: [llms/scripts/run_llms.sh](llms/scripts/run_llms.sh). The LLM outputs are placed in [llms/output](llms/output).

## Evaluation:

- BERTScore: To calculate BERTScore figures, run: [eval/run_bertscore.sh](eval/run_bertscore.sh)
- Transfer accuracy: To run inference on the AA model, follow the instructions in [../authorship_attribution/README.md](../authorship_attribution/README.md). You will need to:
  - Prepare the neutralized data and TST outputs in the format needed for inferencing the AA model. To prepare the data, you run to the data preparation scripts placed in : [../authorship_attribution/Contra-X_AR/prepare_dataset_contraxFormat.sh](../authorship_attribution/Contra-X_AR/prepare_dataset_contraxFormat.sh)
  - Run the TST inference scripts: [../authorship_attribution/Contra-X_AR/run_inference.sh](../authorship_attribution/Contra-X_AR/run_inference.sh)

## Stylometric Analysis:
- To reproduce the stylometric analysis results, run the scripts in: [stylometric_analysis/run_analysis.sh](stylometric_analysis/run_analysis.sh). You will first need to obtain the original [SAMER lexicon](https://aclanthology.org/2020.lrec-1.373.pdf) needed for calculating the readability levels.

