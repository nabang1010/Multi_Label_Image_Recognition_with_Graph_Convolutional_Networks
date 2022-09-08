import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import random
import numpy as np
import argparse
from models_pill import *
from tqdm import tqdm
import wandb
from data_pill import Data_Pill_random_all
import copy

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)

    # add seed 
    parser.add_argument('--seed', type=int, default=42)
    # add weight decay
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--padding_name', type=str,
                        default='_', help='input padding name')
    parser.add_argument('--dim_image', type=int, default=200)
    parser.add_argument('--dim_diagnose', type=int, default=50)
    parser.add_argument('--dim_drugname', type=int, default=50)
    parser.add_argument('--dim_quantity', type=int, default=5)

    
    args = parser.parse_args()
    return args

def config_random(args):
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# import accuarcy score, f1 score, precision score, recall score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Run_model():
    def __init__(self,model_concat, dataloader, optimizer, criterion ):
        self.model_concat = model_concat
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_losses = []

    def train(self, epoch):
        print(" Train epoch _____ {}____".format(epoch))
        self.model_concat.train()
        pred_list, target_list = [], []
        total_loss = 0

        for batch in tqdm(self.dataloader):
            image, diagnose, drugname, quantity, label = batch
            image = image.to('cuda')
            diagnose = diagnose.to('cuda')
            drugname = drugname.to('cuda')
            quantity = quantity.to('cuda')
            label = label.to('cuda')

            self.optimizer.zero_grad()
            output = self.model_concat(image, diagnose, drugname, quantity)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            pred_list += pred.tolist()
            target_list += label.tolist()
            total_loss += loss.item()

        total_loss = round(total_loss / len(self.dataloader) , 3)
        self.train_losses.append(total_loss)
        print(" >>>>>>>>>>>>>>>> Training loss: {}".format(total_loss))

        acc = round(accuracy_score(target_list, pred_list), 3)
        # self.train_acc.append(acc)
        print(" >>>>>>>>>>>>>>>> Training accuracy: {}".format(acc))
        
        f1 = round(f1_score(target_list, pred_list, average='macro'), 3)
        # self.train_f1.append([f1])
        print(" >>>>>>>>>>>>>>>> Training f1: {}".format(f1))
        
        precision = round(precision_score(target_list, pred_list, average='macro'), 3)
        # self.train_precision.append(precision)
        print(" >>>>>>>>>>>>>>>> Training precision: {}".format(precision))
        
        recall = round(recall_score(target_list, pred_list, average='macro'), 3)
        # self.train_recall.append(recall)
        print(" >>>>>>>>>>>>>>>> Training recall: {}".format(recall))


        # store f1_score to text file
        with open('train_f1.txt', 'a') as f:
            f.write(str(f1) + '\n')
        
        # save model to jit torchscript
        if f1 > 0.85:
            self.best_model = copy.deepcopy(self.model_concat)
            # save model to jit torchscript
            self.best_model.eval()
            torch.jit.save(torch.jit.trace(self.best_model, (image, diagnose, drugname, quantity)), '/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/checkpoint/result_pill_concat/model_concat_{}.pt'.format(f1))


        # # log to wandb
        # wandb.log({
        #     "Training loss": total_loss,
        #     "Training accuracy": acc,
        #     "Training f1": f1,
        #     "Training precision": precision,
        #     "Training recall": recall
        # })

if __name__ == '__main__':
    args = get_params()
    config_random(args)

    # wandb.init(project='main_pill', entity='main_pill')
    # wandb.config.update(args)

    # load data
    data_full_train = pd.read_csv("/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/data_XA_full.csv")
    data_emb_train = pd.read_csv("/workspace/nabang1010/LBA_VAIPE/GNN/Multi_Label_Image_Recognition_with_Graph_Convolutional_Networks/data/pill_classify/data_XA_embed_list.csv")

    # create dataset
    train_dataset = Data_Pill_random_all(data_full = data_full_train, data_emb = data_emb_train, status = 'train')
    # test_dataset = Data_Pill(test_data, args.padding_name)

    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # create model
    model_concat = Model_Concat_Pill(args.dim_image, args.dim_diagnose, args.dim_drugname, args.dim_quantity)
    model_concat = model_concat.to('cuda')

    # create optimizer
    optimizer = optim.Adam(model_concat.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # create criterion
    criterion = nn.CrossEntropyLoss()

    # create run model
    run_model = Run_model(model_concat, train_dataloader, optimizer, criterion)

    # train model
    for epoch in range(args.epochs):
        run_model.train(epoch)
        # run_model.test(epoch)
