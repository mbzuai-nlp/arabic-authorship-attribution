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

ckpt_dir = 'exp_data'
    
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
    print(f'R@1 {r1}, R@5 {r5}, R@8 {r8}, R@10 {r10}, MRR {mrr}')
    return r1,r5,r8,r10,mrr

def train_bert(train_dict, test_dic, tqdm_on, model_name, embed_len, id, num_epochs, base_bs, base_lr, mask_classes, coefficient, num_authors, val_dic=None, testBlind_dict=None, valOut_dict=None, testOut_dict=None, testBlindOut_dict=None):
    
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

    # get dataset
    train_x, train_y = train_dict['content'].tolist(), train_dict['Target'].tolist()
    test_x, test_y = test_dic['content'].tolist(), test_dic['Target'].tolist()

    if val_dic is not None:
        val_x, val_y = val_dic['content'].tolist(), val_dic['Target'].tolist()
    if testBlind_dict is not None:
        testBlind_x, testBlind_y = testBlind_dict['content'].tolist(), testBlind_dict['Target'].tolist()
    if valOut_dict is not None:
        valOut_x, valOut_y = valOut_dict['content'].tolist(), valOut_dict['Target'].tolist()
    if testOut_dict is not None:
        testOut_x, testOut_y = testOut_dict['content'].tolist(), testOut_dict['Target'].tolist()
    if testBlindOut_dict is not None:
        testBlindOut_x, testBlindOut_y = testBlindOut_dict['content'].tolist(), testBlindOut_dict['Target'].tolist()

    # training config
    ngpus, dropout = torch.cuda.device_count(), 0.35
    num_tokens, hidden_dim, out_dim = 256, 512, num_authors
    model = BertClassifier(extractor, LogisticRegression(embed_len * num_tokens, hidden_dim, out_dim, dropout=dropout))
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr * ngpus, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_set = BertDataset(train_x, train_y, tokenizer, num_tokens)
    test_set = BertDataset(test_x, test_y, tokenizer, num_tokens)

    if val_dic is not None:
        val_set = BertDataset(val_x, val_y, tokenizer, num_tokens)
    if testBlind_dict is not None:
        testBlind_set = BertDataset(testBlind_x, testBlind_y, tokenizer, num_tokens)
    if valOut_dict is not None:
        valOut_set = BertDataset(valOut_x, valOut_y, tokenizer, num_tokens)
    if testOut_dict is not None:
        testOut_set = BertDataset(testOut_x, testOut_y, tokenizer, num_tokens)
    if testBlindOut_dict is not None:
        testBlindOut_set = BertDataset(testBlindOut_x, testBlindOut_y, tokenizer, num_tokens)

    temperature, sample_unit_size = 0.1, 2
    print(f'coefficient, temperature, sample_unit_size = {coefficient, temperature, sample_unit_size}')

    # logger
    exp_dir = os.path.join(ckpt_dir,
                           f'{id}_{model_name.split("/")[-1]}_coe{coefficient}_temp{temperature}_unit{sample_unit_size}_epoch{num_epochs}_lr{base_lr}')
    writer = SummaryWriter(os.path.join(exp_dir, 'board'))

    # load data
    train_sampler = TrainSamplerMultiClassUnit(train_set, sample_unit_size=sample_unit_size)
    train_loader = DataLoader(train_set, batch_size=base_bs * ngpus, sampler=train_sampler, shuffle=False,
                              num_workers=4 * ngpus, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                             pin_memory=True, drop_last=False)
    if val_dic is not None:
        val_loader = DataLoader(val_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                                pin_memory=True, drop_last=False)
    if testBlind_dict is not None:
        testBlind_loader = DataLoader(testBlind_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                             pin_memory=True, drop_last=False)
    if valOut_dict is not None:
        valOut_loader = DataLoader(valOut_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                                pin_memory=True, drop_last=False)
    if testOut_dict is not None:
        testOut_loader = DataLoader(testOut_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                             pin_memory=True, drop_last=False)
    if testBlindOut_dict is not None:
        testBlindOut_loader = DataLoader(testBlindOut_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                             pin_memory=True, drop_last=False)

    final_test_acc = None
    final_train_preds, final_test_preds = [], []
    best_acc = -1
    best_tv_acc = -1
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # training loop
    for epoch in range(num_epochs):
        train_acc = AverageMeter()
        train_loss = AverageMeter()
        train_loss_1 = AverageMeter()
        train_loss_2 = AverageMeter()
        
        # training
        model.train()
        pg = tqdm(train_loader, leave=False, total=len(train_loader), disable=not tqdm_on)
        for i, (x1, x2, x3, y) in enumerate(pg):  # for x1, x2, x3, y in train_set:
            x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
            pred, feats = model(x, return_feat=True)

            # classification loss
            loss_1 = criterion(pred, y.long())

            # generate the mask
            mask = y.clone().cpu().apply_(lambda x: x not in mask_classes).type(torch.bool).cuda()
            feats, pred, y = feats[mask], pred[mask], y[mask]
            if len(y) == 0:
                continue
            
            # contrastive learning
            sim_matrix = compute_sim_matrix(feats)
            target_matrix = compute_target_matrix(y)
            loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

            # total loss
            loss = loss_1 + coefficient * loss_2

            acc = (pred.argmax(1) == y).sum().item() / len(y)
            train_acc.update(acc,n=len(y))
            train_loss.update(loss.item(),n=len(y))
            train_loss_1.update(loss_1.item(),n=len(y))
            train_loss_2.update(loss_2.item(),n=len(y))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pg.set_postfix({
                'train acc': '{:.6f}'.format(train_acc.avg),
                'train L1': '{:.6f}'.format(train_loss_1.avg),
                'train L2': '{:.6f}'.format(train_loss_2.avg),
                'train L': '{:.6f}'.format(train_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

            # iteration logger
            step = i + epoch * len(pg)
            writer.add_scalar("train-iteration/L1", loss_1.item(), step)
            writer.add_scalar("train-iteration/L2", loss_2.item(), step)
            writer.add_scalar("train-iteration/L", loss.item(), step)
            writer.add_scalar("train-iteration/acc", acc, step)

        print('train acc: {:.6f}'.format(train_acc.avg), 'train L1 {:.6f}'.format(train_loss_1.avg),
              'train L2 {:.6f}'.format(train_loss_2.avg), 'train L {:.6f}'.format(train_loss.avg), f'epoch {epoch}')

        # epoch logger
        writer.add_scalar("train/L1", train_loss_1.avg, epoch)
        writer.add_scalar("train/L2", train_loss_2.avg, epoch)
        writer.add_scalar("train/L", train_loss.avg, epoch)
        writer.add_scalar("train/acc", train_acc.avg, epoch)

        if val_dic is not None:
            model.eval()
            pg = tqdm(val_loader, leave=False, total=len(val_loader), disable=not tqdm_on)
            with torch.no_grad():
                tv_acc = AverageMeter()  # tv stands for train_val
                tv_loss_1 = AverageMeter()
                tv_loss_2 = AverageMeter()
                tv_loss = AverageMeter()
                for i, (x1, x2, x3, y) in enumerate(pg):
                    x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
                    pred, feats = model(x, return_feat=True)
                    
                    # classification
                    #print("y.long()",y.long())
                    loss_1 = criterion(pred, y.long())

                    # contrastive learning
                    sim_matrix = compute_sim_matrix(feats)
                    target_matrix = compute_target_matrix(y)
                    loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

                    # total loss
                    loss = loss_1 + coefficient * loss_2
                        
                    tv_acc.update((pred.argmax(1) == y).sum().item() / len(y),n=len(y))
    
                    tv_loss.update(loss.item(),n=len(y))
                    tv_loss_1.update(loss_1.item(),n=len(y))
                    tv_loss_2.update(loss_2.item(),n=len(y))

                    pg.set_postfix({
                        'train_val acc': '{:.6f}'.format(tv_acc.avg),
                        'epoch': '{:03d}'.format(epoch)
                    })

        # testing
        model.eval()
        pg = tqdm(test_loader, leave=False, total=len(test_loader), disable=not tqdm_on)
        recall_at_1_test=0
        recall_at_5_test=0
        recall_at_8_test=0
        recall_at_10_test=0
        mrr_list=[]
        with torch.no_grad():
            test_acc = AverageMeter()
            test_loss_1 = AverageMeter()
            test_loss_2 = AverageMeter()
            test_loss = AverageMeter()
            for i, (x1, x2, x3, y) in enumerate(pg):
                x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
                pred, feats = model(x, return_feat=True)

                # classification
                loss_1 = criterion(pred, y.long())

                # contrastive learning
                sim_matrix = compute_sim_matrix(feats)
                target_matrix = compute_target_matrix(y)
                loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

                # total loss
                loss = loss_1 + coefficient * loss_2

                ##for e_index in range(base_bs * ngpus):
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

                test_acc.update((pred.argmax(1) == y).sum().item() / len(y),n=len(y))
                test_loss.update(loss.item(),n=len(y))
                test_loss_1.update(loss_1.item(),n=len(y))
                test_loss_2.update(loss_2.item(),n=len(y))

                pg.set_postfix({
                    'test acc': '{:.6f}'.format(test_acc.avg),
                    'epoch': '{:03d}'.format(epoch)
                })

        # logging
        if val_dic is not None:
            writer.add_scalar("tv/L1", tv_loss_1.avg, epoch)
            writer.add_scalar("tv/L2", tv_loss_2.avg, epoch)
            writer.add_scalar("tv/L", tv_loss.avg, epoch)
            writer.add_scalar("tv/acc", tv_acc.avg, epoch)

        mrr=sum(mrr_list)/len(mrr_list)
        writer.add_scalar("test/L1", test_loss_1.avg, epoch)
        writer.add_scalar("test/L2", test_loss_2.avg, epoch)
        writer.add_scalar("test/L", test_loss.avg, epoch)
        writer.add_scalar("test/acc", test_acc.avg, epoch)
        writer.add_scalar("test/R@1", recall_at_1_test/len(mrr_list), epoch)
        writer.add_scalar("test/R@5", recall_at_5_test/len(mrr_list), epoch)
        writer.add_scalar("test/R@8", recall_at_8_test/len(mrr_list), epoch)
        writer.add_scalar("test/R@10", recall_at_10_test/len(mrr_list), epoch)
        writer.add_scalar("test/MRR", mrr, epoch)

        scheduler.step()

        #Reporting Accuracy, R@5, R@8, R@10, MRR on all evaluation sets
        all_results=[]
        if val_dic is not None:
            print("Evaluating Dev_in:")
            pg_valBlind = tqdm(val_loader, leave=False, total=len(val_loader), disable=not tqdm_on)
            val_r1,r5,r8,r10,mrr=eval_model(pg_valBlind,model)
            all_results.extend([val_r1,r5,r8,r10,mrr])

        print("Evaluating Test_in:")
        pg_test = tqdm(test_loader, leave=False, total=len(test_loader), disable=not tqdm_on)
        r1,r5,r8,r10,mrr=eval_model(pg_test,model)
        all_results.extend([r1,r5,r8,r10,mrr])

        if testBlind_dict is not None:
            print("Evaluating TestBlind_in:")
            pg_testBlind = tqdm(testBlind_loader, leave=False, total=len(testBlind_loader), disable=not tqdm_on)
            r1,r5,r8,r10,mrr=eval_model(pg_testBlind,model)
            all_results.extend([r1,r5,r8,r10,mrr])

        if valOut_dict is not None:
            print("Evaluating Dev_out:")
            pg_valOutBlind = tqdm(valOut_loader, leave=False, total=len(valOut_loader), disable=not tqdm_on)
            valOut_r1,r5,r8,r10,mrr=eval_model(pg_valOutBlind,model)
            all_results.extend([valOut_r1,r5,r8,r10,mrr])
        
        if testOut_dict is not None:
            print("Evaluating Test_out:")
            pg_testOutBlind = tqdm(testOut_loader, leave=False, total=len(testOut_loader), disable=not tqdm_on)
            r1,r5,r8,r10,mrr=eval_model(pg_testOutBlind,model)
            all_results.extend([r1,r5,r8,r10,mrr])
        
        if testBlindOut_dict is not None:
            print("Evaluating TestBlind_out:")
            pg_testBlindOutBlind = tqdm(testBlindOut_loader, leave=False, total=len(testBlindOut_loader), disable=not tqdm_on)
            r1,r5,r8,r10,mrr=eval_model(pg_testBlindOutBlind,model)
            all_results.extend([r1,r5,r8,r10,mrr])

        print("All Results:")
        print('\t'.join([str(x) for x in all_results]))
        
        final_test_acc = test_acc.avg
        
        # save model
        if val_dic is None: #This should be used for experiments using only train and test, as original code
            if test_acc.avg:
                if test_acc.avg >= best_acc:
                    cur_models = os.listdir(exp_dir)
                    for cur_model in cur_models:
                        if cur_model.endswith(".pt"):
                            os.remove(os.path.join(exp_dir, cur_model))
                    save_model(exp_dir, f'{id}_val{final_test_acc:.5f}_e{epoch}.pt', model)
            best_acc = max(best_acc, test_acc.avg)        
        else: #We choose checkpoint according to the tuning set
            print(f'epoch {epoch}, train val acc {tv_acc.avg}')
            print("======> check")
            print(val_r1,tv_acc.avg)
            #if condition added
            if valOut_dict is not None:
                val_acc=(tv_acc.avg+valOut_r1)/2 #calculating macro-average for in-documnet and cross-document accuracies
                print("Macro avg val acc:",val_acc)
            else:
                val_acc=tv_acc.avg
            final_tv_acc = val_acc
            if val_acc >= best_tv_acc:
                cur_models = os.listdir(exp_dir)
                for cur_model in cur_models:
                    if cur_model.endswith("_validation.pt"):
                        os.remove(os.path.join(exp_dir, cur_model))
                save_model(exp_dir, f'{id}_val{val_acc:.5f}_e{epoch}_validation.pt', model)
                best_tv_acc = val_acc

    # save checkpoint
    save_model(exp_dir, f'{id}_val{final_test_acc:.5f}_finale{epoch}.pt', model)

    print(
        f'Training complete after {num_epochs} epochs. Final val acc = {final_tv_acc}, '
        f'best val acc = {best_tv_acc}, best test acc = {best_acc}.'
        f'Final test acc {final_test_acc}')

    return final_test_acc, final_train_preds, final_test_preds
