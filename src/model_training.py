import argparse
import pandas as pd
import numpy as np
import random
import transformers
import torch
import torch.nn.functional as F
import warnings

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from time import time
from torch import nn, optim
from transformers import AutoTokenizer, AdamW

from transformer_model import TransformerTBSA
from train import train_epoch
from eval import eval_model, get_predictions
from data_loader import create_data_loader


def run_epoch(args, epoch):
    print(f'Epoch {epoch + 1}/{args.epochs}')
    print('-' * 10)
    train_acc, train_loss, train_avg_losses, avg_accs = train_epoch(args)
    
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(args)

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    args.history['train_acc'].append(train_acc)
    args.history['train_loss'] += train_avg_losses
    args.history['val_acc'].append(val_acc)
    args.history['val_loss'].append(val_loss)

    if val_acc > args.best_acc:
        if args.out_model is None:
            args.out_model = 'models/' + 'transformer_' + str(time()) + '.pt' 
        torch.save(args.model.state_dict(), args.out_model)
        args.best_acc = val_acc
        print('NEW BEST ACCURACY:', args.best_acc)
        print('NEW BEST ARGS: ', args.batch_size, args.lr, args.final_dropout)


def parse_args():
    parser = argparse.ArgumentParser("SploilerAlert")
    parser.add_argument('data', type=str, help='path to data file')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        help='Huggingface Transformers pretrained model. distilbert-base-uncased by default')
    parser.add_argument('--in_model', default=None, help="Path to the model weights to load for training/eval. \n" +
                        "Must conform with the chosen architecture")
    parser.add_argument('--out_model', default=None, help="Name of the trained model (will be saved with that name). Used for training only")
    parser.add_argument("--max_len", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences "
                        "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--acc_steps',
                        type=int,
                        default=1,
                        help="gradient accumulation steps")
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help="number of training epochs")
    parser.add_argument('--test',
                        default=False,
                        action='store_true',
                        help="Whether on test mode")

    return parser.parse_args()

def model_train(args):
    args.loss_fn = nn.CrossEntropyLoss().to(args.device)
    args.model = TransformerTBSA(args)
    args.model.to(args.device)
    args.optimizer = AdamW(args.model.parameters(), lr=args.lr, correct_bias=False)
    args.history = defaultdict(list)
    for epoch in range(args.epochs):
        run_epoch(args, epoch)


def model_eval(args):
    args.model = TransformerTBSA(args)
    if args.in_model:
        args.model.load_state_dict(torch.load(args.in_model))
    args.model.to(args.device)

    args.test_dl = create_data_loader(df, args)        
    y_review_texts, y_pred, y_test = get_predictions(args)
    print(accuracy_score(y_test, y_pred))
    print(f1_score(y_test, y_pred, average='macro'))
    print(confusion_matrix(y_test, y_pred))


def main():
    args = parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data, sep='\t')
    df.sentiment += 1
    df['target_text'] = df.apply(lambda x: x.text + ' [SEP] ' + x.target, axis=1)

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ### Model Evaluation
    if args.test:
        model_eval(args)
    ### Model Training, with some hyperparam tuning
    else:
        df_train, df_eval = train_test_split(df, test_size=0.2, random_state=args.seed)
        print(df_train.shape, df_eval.shape)
        args.train_size = len(df_train)
        args.eval_size = len(df_eval)
        args.best_acc = 0
        args.final_dropout = 0.2
        args.lr = 5e-5
        best_batch_size =  8
        best_acc = 0
        for bs in [8, 32, 128]:
            args.batch_size = bs

            args.train_dl = create_data_loader(df_train, args)
            args.eval_dl = create_data_loader(df_eval, args)
            model_train(args)
            if best_acc < args.best_acc:
                best_acc = args.best_acc
                best_batch_size = bs

        #args.batch_size = best_batch_size
        #for lr in [5e-4, 5e-3]:
        #    for drop in [0.2, 0.5]:
        #        args.final_dropout = drop
        #        args.lr = lr
        #        model_train(args)


if __name__ == "__main__":
    main()