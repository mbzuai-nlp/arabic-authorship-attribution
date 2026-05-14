a3d_corpus_dir=../../../A3D_corpus
style_neutralization_dir=../../style_neutralization

python sample_author_references.py ${a3d_corpus_dir}/dataset_A3D.csv ${style_neutralization_dir}/data/dataset_A3Dtest_neutralized_mt.csv ../data/dataset_A3Dtest_neutralized_mt_wTrainExamples.csv
