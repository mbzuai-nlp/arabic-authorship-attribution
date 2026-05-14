
a3d_corpus_dir=../../../A3D_corpus

# We provide the output of our style neutralization scripts. You need the dataset_A3D.csv data to unmuddle the files using the Muddler Tool (https://github.com/CAMeL-Lab/muddler):
for approach in rewrite mt; do
  muddler unmuddle -s ${a3d_corpus_dir}/dataset_A3D.csv -m ../data/dataset_A3Dtest_neutralized_${approach}.muddled ../data/dataset_A3Dtest_neutralized_${approach}.csv
done

#To generate your own files, run:

nohup python neutralize_style.py ${a3d_corpus_dir}/dataset_A3D.csv ../data/dataset_A3Dtest_neutralized_rewrite.csv rewrite > rewrite.log &

nohup python neutralize_style.py ${a3d_corpus_dir}/dataset_A3D.csv ../data/dataset_A3Dtest_neutralized_mt.csv mt > mt.log &

