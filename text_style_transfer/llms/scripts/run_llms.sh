
data_input=../data/dataset_A3Dtest_neutralized_mt_wTrainExamples.csv

mkdir -p ../logs

nohup python generate_style_deepseek.py ${data_input} ../output/tst_deepseek.csv > ../logs/deepseek.log &

nohup python generate_style_fanar.py ${data_input} ../output/tst_fanar.csv > ../logs/fanar.log &

nohup python generate_style_gemini.py ${data_input} ../output/tst_gemini.csv > ../logs/gemini.log &

nohup python generate_style_gpt.py ${data_input} ../output/tst_gpt_2.csv > ../logs/gpt.log &

nohup python generate_style_jais.py ${data_input} ../output/tst_jais.csv > ../logs/jais.log &

nohup python generate_style_llama.py ${data_input} ../output/tst_llama.csv > ../logs/llama.log &

# Processing the output of the LLMs to:
# (1) obtain the TST generation from a formated LLM output
# (2) place the neutralized text for a snippet in case the LLM was unable to provide a TST generation (this happens in 28 cases for gemini and 53 for Jais)
# The output is placed under ../output with "_processed" suffix attached to the files
for model in gpt deepseek gemini llama fanar jais; do
  python process_generated_tst_output.py ../output/tst_${model}.csv
done