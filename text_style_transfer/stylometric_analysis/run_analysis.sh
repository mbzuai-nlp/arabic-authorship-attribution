

mkdir -p tmp_files
mkdir -p feature_vectors

SAMER_LEX_DIR=/Users/injy.hamed/Documents/Projects/Arabic_AA_TST/SAMER_lexicon
samer_lex=${SAMER_LEX_DIR}/samer-readability-lexicon/samer-readability-lexicon-v2/SAMER-Readability-Lexicon-v2.tsv
data_neutralized_wAuthorRef=../llms/data/dataset_A3Dtest_neutralized_mt_wTrainExamples.csv
data_neutralized=../style_neutralization/data/dataset_A3Dtest_neutralized_mt.csv

#Obtaining feature vectors for author reference snippets:
python get_aldi.py ${data_neutralized_wAuthorRef} tmp_files/aldi_authorRef.csv content_example

python get_readability_levels.py ${data_neutralized_wAuthorRef} tmp_files/readability_levels_authorRef.csv content_example

python measure_stylometric_features.py ${data_neutralized_wAuthorRef} ${samer_lex} tmp_files/readability_levels_authorRef.csv tmp_files/aldi_authorRef.csv content_example feature_vectors/features_vector_authorRef

python measure_stylometric_features.py ${data_neutralized_wAuthorRef} ${samer_lex} tmp_files/readability_levels_authorRef.csv tmp_files/aldi_authorRef.csv content_example feature_vectors/features_vector_authorRef

#Obtaining feature vectors for LLM TST outputs:
for model in gpt gemini fanar jais deepseek llama; do
  echo $model
  python get_readability_levels.py ../llms/output/tst_${model}_processed.csv tmp_files/readability_levels_${model}.csv content_styled
done

for model in gpt gemini fanar jais deepseek llama; do
  echo $model
  python get_aldi.py ../llms/output/tst_${model}_processed.csv tmp_files/aldi_${model}.csv content_styled
done

for model in gpt gemini fanar jais deepseek llama; do
  echo $model
  python measure_stylometric_features.py ../llms/output/tst_${model}_processed.csv ${samer_lex} tmp_files/readability_levels_${model}.csv tmp_files/aldi_${model}.csv content_styled feature_vectors/features_vector_${model}
done

for model in gpt gemini fanar jais deepseek llama; do
  echo $model
  python cal_cosSim_features.py feature_vectors/features_vector_authorRef feature_vectors/features_vector_${model} tmp_files/cosSim_authorRef_${model} ${data_neutralized_wAuthorRef}
done


#Obtaining feature vectors for style neutralized texts:
python get_readability_levels.py ${data_neutralized} tmp_files/readability_levels_neutralized_mt.csv content

python get_aldi.py ${data_neutralized} tmp_files/aldi_neutralized_mt.csv  content

python measure_stylometric_features.py ${data_neutralized} ${samer_lex} tmp_files/readability_levels_neutralized_mt.csv tmp_files/aldi_neutralized_mt.csv content feature_vectors/features_vector_neutralized_mt

python cal_cosSim_features.py feature_vectors/features_vector_authorRef feature_vectors/features_vector_neutralized_mt tmp_files/cosSim_reference_mt ${data_neutralized_wAuthorRef}
