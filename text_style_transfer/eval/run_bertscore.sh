llm_processed_files_dir=../llms/output
style_neutral_files_dir=../style_neutralization/data
a3d_corpus_dir=../../A3D_corpus

python calculate_bertscore.py ${llm_processed_files_dir}/tst_gemini_processed.csv ${a3d_corpus_dir}/dataset_A3D.csv content_styled

python calculate_bertscore.py ${llm_processed_files_dir}/tst_fanar_processed.csv ${a3d_corpus_dir}/dataset_A3D.csv content_styled

python calculate_bertscore.py ${llm_processed_files_dir}/tst_jais_processed.csv ${a3d_corpus_dir}/dataset_A3D.csv content_styled

python calculate_bertscore.py ${llm_processed_files_dir}/tst_gpt_processed.csv ${a3d_corpus_dir}/dataset_A3D.csv content_styled

python calculate_bertscore.py ${llm_processed_files_dir}/tst_deepseek_processed.csv ${a3d_corpus_dir}/dataset_A3D.csv content_styled

python calculate_bertscore.py ${llm_processed_files_dir}/tst_llama_processed.csv ${a3d_corpus_dir}/dataset_A3D.csv content_styled

python calculate_bertscore.py ${style_neutral_files_dir}/dataset_A3Dtest_neutralized_mt.csv ${a3d_corpus_dir}/dataset_A3D.csv content

python calculate_bertscore.py ${style_neutral_files_dir}/dataset_A3Dtest_neutralized_rewrite.csv ${a3d_corpus_dir}/dataset_A3D.csv content
