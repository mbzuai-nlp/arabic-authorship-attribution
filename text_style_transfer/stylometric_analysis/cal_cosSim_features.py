import sys
import csv
import torch

def main():
    cos_fn = torch.nn.CosineSimilarity(dim=0)
    list_feature_dimensions=["","_surfaceFeatures","_syntacticFeatures","_lexicalFeatures","_langFeatures"]
    input_corpus_csv_file=open(sys.argv[4], newline='')
    csvreader = csv.DictReader(input_corpus_csv_file)
    dict_corpus={}
    index=0

    for row in csvreader:
        dict_corpus[index]=row['corpus']
        index+=1

    for feature_dimension in list_feature_dimensions:
        tensor_pos_values_ref=torch.load(sys.argv[1]+feature_dimension)
        tensor_pos_values_tst=torch.load(sys.argv[2]+feature_dimension)
        output_file=open(sys.argv[3]+feature_dimension,'w')
        list_cos=[]
        list_cos_Hindawi=[]
        list_cos_Gumar=[]
        for index in range(len(tensor_pos_values_ref)):
            tensor_pos_values_ref_i=tensor_pos_values_ref[index]
            tensor_pos_values_tst_i=tensor_pos_values_tst[index]
            cos_sim = cos_fn(tensor_pos_values_ref_i, tensor_pos_values_tst_i)
            list_cos.append(cos_sim.item())
            if dict_corpus[index]=='Hindawi':
                list_cos_Hindawi.append(cos_sim.item())
            elif dict_corpus[index]=='Gumar':
                list_cos_Gumar.append(cos_sim.item())
            output_file.write(str(cos_sim.item()))
        print("feature_dimension:",feature_dimension)
        print("overall:",sum(list_cos)/len(list_cos))
        print("Hindawi:",sum(list_cos_Hindawi)/len(list_cos_Hindawi))
        print("Gumar:",sum(list_cos_Gumar)/len(list_cos_Gumar))

if __name__=="__main__":
    main()