
#Running inference on the original A3D dataset
model=CAMeL-Lab/bert-base-arabic-camelbert-mix

python main_inference.py --dataset a3d --id a3d_inference --tqdm True --model ${model} \
    --model_path exp_data/hindawi_gumarRaw_setDocs_setup3_200_camelbertMix_fixed_val0.90167_e8_validation_bkup.pt
    


#Running inference on the LLM TST outputs
python main_inference.py --dataset a3d_tst_gpt --id a3d_tst_gpt --tqdm True --model CAMeL-Lab/bert-base-arabic-camelbert-mix \
    --model_path exp_data/hindawi_gumarRaw_setDocs_setup3_200_camelbertMix_fixed_val0.90167_e8_validation_bkup.pt
    
python main_inference.py --dataset a3d_tst_deepseek --id a3d_tst_deepseek --tqdm True --model CAMeL-Lab/bert-base-arabic-camelbert-mix \
    --model_path exp_data/hindawi_gumarRaw_setDocs_setup3_200_camelbertMix_fixed_val0.90167_e8_validation_bkup.pt
    
python main_inference.py --dataset a3d_tst_gemini --id a3d_tst_gemini --tqdm True --model CAMeL-Lab/bert-base-arabic-camelbert-mix \
    --model_path exp_data/hindawi_gumarRaw_setDocs_setup3_200_camelbertMix_fixed_val0.90167_e8_validation_bkup.pt
    
python main_inference.py --dataset a3d_tst_llama --id a3d_tst_llama --tqdm True --model CAMeL-Lab/bert-base-arabic-camelbert-mix \
    --model_path exp_data/hindawi_gumarRaw_setDocs_setup3_200_camelbertMix_fixed_val0.90167_e8_validation_bkup.pt
    
python main_inference.py --dataset a3d_tst_fanar --id a3d_tst_fanar --tqdm True --model CAMeL-Lab/bert-base-arabic-camelbert-mix \
    --model_path exp_data/hindawi_gumarRaw_setDocs_setup3_200_camelbertMix_fixed_val0.90167_e8_validation_bkup.pt
    
python main_inference.py --dataset a3d_tst_jais --id a3d_tst_jais --tqdm True --model CAMeL-Lab/bert-base-arabic-camelbert-mix \
    --model_path exp_data/hindawi_gumarRaw_setDocs_setup3_200_camelbertMix_fixed_val0.90167_e8_validation_bkup.pt
    
python main_inference.py --dataset a3d_tst_neutral_rewrite --id a3d_tst_neutral_rewrite --tqdm True --model CAMeL-Lab/bert-base-arabic-camelbert-mix \
    --model_path exp_data/hindawi_gumarRaw_setDocs_setup3_200_camelbertMix_fixed_val0.90167_e8_validation_bkup.pt
    
python main_inference.py --dataset a3d_tst_neutral_mt --id a3d_tst_neutral_mt --tqdm True --model CAMeL-Lab/bert-base-arabic-camelbert-mix \
    --model_path exp_data/hindawi_gumarRaw_setDocs_setup3_200_camelbertMix_fixed_val0.90167_e8_validation_bkup.pt
    
