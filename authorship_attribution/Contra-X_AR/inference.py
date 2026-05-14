import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from contrax_loss import compute_sim_matrix, compute_target_matrix, contrastive_loss
from dataset import BertDataset
from dataset import TrainSamplerMultiClassUnit
from models import BertClassifier
from models import LogisticRegression
from utils import *  
import torch.nn.functional as F
    
def check_recall_at_k(L,index,k):
    L_copy=L.tolist()
    L_copy.sort(reverse=True)
    value_at_k=L_copy[k-1]
    return L[index]>=value_at_k

def get_MRR(L,index):
    L_copy=L.tolist()
    L_copy.sort(reverse=True)  
    rank=L_copy.index(L[index])+1
    return rank

def eval_model(pg,model):
    recall_at_1_test=0
    recall_at_5_test=0
    recall_at_8_test=0
    recall_at_10_test=0
    mrr_list=[]
    with torch.no_grad():
        for i, (x1, x2, x3, y) in enumerate(pg):
            x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
            pred, feats = model(x, return_feat=True)

            for e_index in range(len(y)):
                y_label=y[e_index]
                softmax_pred=F.softmax(pred[e_index])
                if check_recall_at_k(softmax_pred,y_label,1):
                    recall_at_1_test+=1
                if check_recall_at_k(softmax_pred,y_label,5):
                    recall_at_5_test+=1
                if check_recall_at_k(softmax_pred,y_label,8):
                    recall_at_8_test+=1
                if check_recall_at_k(softmax_pred,y_label,10):
                    recall_at_10_test+=1
                mrr_list.append(get_MRR(softmax_pred,y_label))
    r1=recall_at_1_test/len(mrr_list)
    r5=recall_at_5_test/len(mrr_list)
    r8=recall_at_8_test/len(mrr_list)
    r10=recall_at_10_test/len(mrr_list)
    mrr=sum(mrr_list)/len(mrr_list)
    print("len(mrr_list)",len(mrr_list))
    print(f'R@1 {r1}, R@5 {r5}, R@8 {r8}, R@10 {r10}, MRR {mrr}')
    return r1,r5,r8,r10,mrr

def inference_bert(train_dict, val_dic, test_dic, testBlind_dict, valOut_dict, testOut_dict, testBlindOut_dict, tqdm_on, model_name, model_path, embed_len, id, base_bs, num_authors):
        
    # tokenizer and pretrained model
    tokenizer, extractor = None, None
    if 'bert-base' in model_name or 'ALDi' in model_name:
        print("Using BertTokenizer")
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained(model_name)
        extractor = BertModel.from_pretrained(model_name)
    elif 'deberta-base' in model_name:
        print("Using DebertaTokenizer")
        from transformers import DebertaTokenizer, DebertaModel
        tokenizer = DebertaTokenizer.from_pretrained(model_name)
        extractor = DebertaModel.from_pretrained(model_name)
    else:
        print("Using AutoTokenizer")
        from transformers import AutoTokenizer,AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        extractor = AutoModel.from_pretrained(model_name)

    # update extractor
    for param in extractor.parameters():
        param.requires_grad = True

    # training config
    ngpus, dropout = torch.cuda.device_count(), 0.35
    num_tokens, hidden_dim, out_dim = 256, 512, num_authors
    model = BertClassifier(extractor, LogisticRegression(embed_len * num_tokens, hidden_dim, out_dim, dropout=dropout))
    model = nn.DataParallel(model).cuda()

    #loading model
    state_dict_loaded = torch.load(model_path)
    state_dict={}
    for k,v in state_dict_loaded.items():
        state_dict["module."+k]=v
    model.load_state_dict(state_dict)
    
    # testing
    model.eval()

    if len(testBlind_dict)>0:
        print("Evaluating TestBlind_in:")
        testBlind_x, testBlind_y = testBlind_dict['content'].tolist(), testBlind_dict['Target'].tolist()
        testBlind_set = BertDataset(testBlind_x, testBlind_y, tokenizer, num_tokens)
        testBlind_loader = DataLoader(testBlind_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus, pin_memory=True, drop_last=False)
        pg_testBlind = tqdm(testBlind_loader, leave=False, total=len(testBlind_loader), disable=not tqdm_on)
        r1,r5,r8,r10,mrr=eval_model(pg_testBlind,model)

    if len(testBlindOut_dict)>0:
        print("Evaluating TestBlind_out:")
        testBlindOut_x, testBlindOut_y = testBlindOut_dict['content'].tolist(), testBlindOut_dict['Target'].tolist()
        testBlindOut_set = BertDataset(testBlindOut_x, testBlindOut_y, tokenizer, num_tokens)
        testBlindOut_loader = DataLoader(testBlindOut_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus, pin_memory=True, drop_last=False)
        pg_testBlindOutBlind = tqdm(testBlindOut_loader, leave=False, total=len(testBlindOut_loader), disable=not tqdm_on)
        r1,r5,r8,r10,mrr=eval_model(pg_testBlindOutBlind,model)
    
    if len(val_dic)>0:
        print("Evaluating Dev_in:")
        val_x, val_y = val_dic['content'].tolist(), val_dic['Target'].tolist()
        val_set = BertDataset(val_x, val_y, tokenizer, num_tokens)
        val_loader = DataLoader(val_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus, pin_memory=True, drop_last=False)
        pg_valBlind = tqdm(val_loader, leave=False, total=len(val_loader), disable=not tqdm_on)
        r1,r5,r8,r10,mrr=eval_model(pg_valBlind,model)

    if len(test_dic)>0:
        print("Evaluating Test_in:")
        test_x, test_y = test_dic['content'].tolist(), test_dic['Target'].tolist()
        test_set = BertDataset(test_x, test_y, tokenizer, num_tokens)
        test_loader = DataLoader(test_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus, pin_memory=True, drop_last=False)
        pg_test = tqdm(test_loader, leave=False, total=len(test_loader), disable=not tqdm_on)
        r1,r5,r8,r10,mrr=eval_model(pg_test,model)
    
    if len(valOut_dict)>0:    
        print("Evaluating Dev_out:")
        valOut_x, valOut_y = valOut_dict['content'].tolist(), valOut_dict['Target'].tolist()
        valOut_set = BertDataset(valOut_x, valOut_y, tokenizer, num_tokens)
        valOut_loader = DataLoader(valOut_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus, pin_memory=True, drop_last=False)
        pg_valOutBlind = tqdm(valOut_loader, leave=False, total=len(valOut_loader), disable=not tqdm_on)
        r1,r5,r8,r10,mrr=eval_model(pg_valOutBlind,model)

    if len(testOut_dict)>0:
        print("Evaluating Test_out:")
        testOut_x, testOut_y = testOut_dict['content'].tolist(), testOut_dict['Target'].tolist()
        testOut_set = BertDataset(testOut_x, testOut_y, tokenizer, num_tokens)
        testOut_loader = DataLoader(testOut_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus, pin_memory=True, drop_last=False)
        pg_testOutBlind = tqdm(testOut_loader, leave=False, total=len(testOut_loader), disable=not tqdm_on)
        r1,r5,r8,r10,mrr=eval_model(pg_testOutBlind,model)

