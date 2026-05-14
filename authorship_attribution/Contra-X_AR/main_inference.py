import argparse
import os
import random
import warnings

import numpy
import torch

from inference import inference_bert
from utils import load_dataset_dataframe, build_train_test

random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    datasets = ['imdb62', 'blog', 'turing','a3d','a3d_tst_neutral_rewrite','a3d_tst_neutral_mt','a3d_tst_gemini','a3d_tst_gpt','a3d_tst_deepseek','a3d_tst_llama','a3d_tst_fanar','a3d_tst_jais']
    parser = argparse.ArgumentParser(description=f'Training models for datasets {datasets}')
    parser.add_argument('--dataset', type=str, help='dataset used for training', choices=datasets)
    parser.add_argument('--id', type=str, default='0', help='experiment id')
    parser.add_argument('--gpu', type=str, help='the cuda devices used for training', default="0,1,2,3")
    parser.add_argument('--tqdm', type=bool, help='whether tqdm is on', default=False)
    parser.add_argument('--authors', type=int, help='number of authors', default=None)
    parser.add_argument('--samples-per-auth', type=int, help='number of samples per author', default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model', type=str, default='microsoft/deberta-base')
    parser.add_argument('--model_path', type=str, help='path of pretrained model')
    parser.add_argument('--coe', type=float, default=1)

    # dataset - num of authors mapping
    default_num_authors = {
        'a3d':40,
        'a3d_tst_neutral_rewrite':40,
        'a3d_tst_neutral_mt':40,
        'a3d_tst_gemini':40,
        'a3d_tst_gpt':40,
        'a3d_tst_deepseek':40,
        'a3d_tst_llama':40,
        'a3d_tst_fanar':40,
        'a3d_tst_jais':40
    }

    # parse args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = args.dataset
    num_authors = args.authors if args.authors is not None else default_num_authors[args.dataset]
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))  # print all args

    # masked classes
    mask_classes = {
        'a3d':[],
        'a3d_tst_neutral_rewrite':[],
        'a3d_tst_neutral_mt':[],
        'a3d_tst_gemini':[],
        'a3d_tst_gpt':[],
        'a3d_tst_deepseek':[],
        'a3d_tst_llama':[],
        'a3d_tst_fanar':[],
        'a3d_tst_jais':[]
    }

    # load data and remove emails containing the sender's name
    df = load_dataset_dataframe(source)

    if args.authors is not default_num_authors[args.dataset]:
        warnings.warn(f"Number of authors for dataset {args.dataset} is {default_num_authors[args.dataset]}, "
                      f"but got {args.authors} instead. ")

    if args.samples_per_auth is not None:
        warnings.warn(f"Number of samples per author specified as {args.samples_per_auth}, which is a "
                      f"dangerous argument. ")

    limit = num_authors
    print("Number of authors: ", limit)
    nlp_train, nlp_val, nlp_test, nlp_testBlind, nlp_valOut, nlp_testOut, nlp_testBlindOut = build_train_test(df, source, limit, per_author=args.samples_per_auth, seed=0)

    if source.startswith('a3d'):
        inference_bert(nlp_train, nlp_val, nlp_test, nlp_testBlind, nlp_valOut, nlp_testOut, nlp_testBlindOut, args.tqdm, args.model, args.model_path, 768, args.id, base_bs=7, num_authors=num_authors)
