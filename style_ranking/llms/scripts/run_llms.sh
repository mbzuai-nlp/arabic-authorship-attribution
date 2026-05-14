
data_input=../../style_ranking_data.csv

python rank_style_deepseek.py ${data_input} ../output/style_ranking_output_deepseek

python rank_style_fanar.py ${data_input} ../output/style_ranking_output_fanar

python rank_style_gemini.py ${data_input} ../output/style_ranking_output_gemini

python rank_style_gpt.py ${data_input} ../output/style_ranking_output_gpt

python rank_style_jais.py ${data_input} ../output/style_ranking_output_jais

python rank_style_llama.py ${data_input} ../output/style_ranking_output_llama
