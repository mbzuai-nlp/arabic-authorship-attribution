

python get_styleRankings.py \
    --input_data  ../../style_ranking/style_ranking_data.csv \
    --output_embeddings embeddings.tmp \
    --output style_ranking_output_AA \
    --model CAMeL-Lab/bert-base-arabic-camelbert-mix\
    --model_path exp_data/hindawi_gumarRaw_setDocs_setup3_200_camelbertMix_fixed_val0.90167_e8_validation_bkup.pt
    
rm embeddings.tmp