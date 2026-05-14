
mkdir -p datasets

python prepare_dataset_contraxFormat.py ../../A3D_corpus/dataset_A3D.csv datasets/dataset_A3D_contraxFormat.csv

#Preparing the inference files for the LLM TST generations

llm_processed_files_dir=../../text_style_transfer/llms/output

for model in gpt deepseek gemini llama fanar jais; do
  python prepare_tst_contraxFormat.py ${llm_processed_files_dir}/tst_${model}_processed.csv content_styled datasets/tst_${model}_contraxFormat2.csv
done

#Preparing the inference files for the style neutralization outputs

style_neutralized_files_dir=../../text_style_transfer/style_neutralization/data

for approach in mt rewrite; do
  python prepare_tst_contraxFormat.py ${style_neutralized_files_dir}/dataset_A3Dtest_neutralized_${approach}.csv content datasets/dataset_A3Dtest_neutralized_${approach}_contraxFormat2.csv
done