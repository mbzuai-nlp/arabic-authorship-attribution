mkdir -p logs

nohup python main.py --dataset a3d --id camelbertMix --gpu 0 --tqdm True --epochs 10 --model CAMeL-Lab/bert-base-arabic-camelbert-mix > logs/log_camelbertMix&

nohup python main.py --dataset a3d --id camelbertMSA --gpu 0 --tqdm True --epochs 10 --model CAMeL-Lab/bert-base-arabic-camelbert-msa > logs/log_camelbertMSA&

nohup python main.py --dataset a3d --id camelbertMSADID --gpu 0 --tqdm True --epochs 10 --model CAMeL-Lab/bert-base-arabic-camelbert-msa-did-nadi > logs/log_camelbertMSADID&

nohup python main.py --dataset a3d --id camelbertMixDID --gpu 0 --tqdm True --epochs 10 --model CAMeL-Lab/bert-base-arabic-camelbert-mix-did-nadi > logs/log_camelbertMixDID&

nohup python main.py --dataset a3d --id arabert --gpu 0 --tqdm True --epochs 10 --model aubmindlab/bert-base-arabert > logs/log_arabert&

nohup python main.py --dataset a3d --id aldi --gpu 0 --tqdm True --epochs 10 --model AMR-KELEG/Sentence-ALDi > logs/log_aldi&

nohup python main.py --dataset a3d --id mbert --gpu 0 --tqdm True --epochs 10 --model google-bert/bert-base-multilingual-cased > logs/log_mbert&

nohup python main.py --dataset a3d --id mdeberta --gpu 0 --tqdm True --epochs 10 --model microsoft/mdeberta-v3-base > logs/log_mdeberta&

