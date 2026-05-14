# Authorship Attribution Task
This directory contains scripts and resources for reproducing the authorship attribution experiments reported in our paper.

## Zero-shot Setting:
- To reproduce the zero-shot results, run: [zero_shot/run_zeroshot_eval.sh](zero_shot/run_zeroshot_eval.sh).

 ## Contra-X_AR:
This directory contains our modified code for [Contra-X](https://github.com/BoAi01/Contra-X) github repo. 
- Installation: 
  - You need to create a conda environment and install the packages in requirements.txt:
```bash
cd Contra-X_AR
conda create -n contrax python=3.10
conda activate contrax
pip install -r requirements.txt 
```
- Data Preparation: 
  - You need to prepare the original A3D dataset, neutralized data and TST outputs in the format needed for training and inferencing the AA model. To prepare the data, run: [Contra-X_AR/prepare_dataset_contraxFormat.sh](Contra-X_AR/prepare_dataset_contraxFormat.sh). The output is placed under [Contra-X_AR/datasets](Contra-X_AR/datasets). 
- Training: 
  - To the train the AA models, run: [Contra-X_AR/run_training.sh](Contra-X_AR/run_training.sh)
- Inference: 
  - To run inference on the AA models for the A3D corpus and the LLM TST outputs, run the scripts in: [Contra-X_AR/run_inference.sh](Contra-X_AR/run_inference.sh)