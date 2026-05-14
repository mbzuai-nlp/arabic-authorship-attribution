
model=google-bert/bert-base-multilingual-cased
model=CAMeL-Lab/bert-base-arabic-camelbert-msa
model=CAMeL-Lab/bert-base-arabic-camelbert-mix
model=CAMeL-Lab/bert-base-arabic-camelbert-msa-did-nadi
model=CAMeL-Lab/bert-base-arabic-camelbert-mix-did-nadi
model=aubmindlab/bert-base-arabert
model=AMR-KELEG/Sentence-ALDi
model=microsoft/mdeberta-v3-base

python eval_zeroshot.py ../../A3D_corpus/dataset_A3D.csv ${model}
