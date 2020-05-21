import yaml
import argparse
import torch
import numpy as np
import sys
sys.path.append("..")
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from grammars.base import LegoGram
from grammars.rnn_sampler import LegoGramRNNSampler
from model import RNN
import pandas  as pd
from providers import MolecularNotationDataset, SpecialTokenWrapperModel, SmilesModel, robust_standardizer
#from legogram.apps import LegoGramRNNSampler
from utils import collate_fn, collect
import os
from torch.utils.tensorboard import SummaryWriter
import json
from torch.nn import MSELoss
from copy import deepcopy

def train(args):
    if args.create_dataset:
        df = pd.read_csv("../data/endpoints_calculated_std.csv")
        smiles = df["smiles"].to_list()
        data = df[df.columns[3:]].to_numpy()
        print("Building LegoModel")
        legoModel = LegoGram(smiles = smiles, nworkers=8)
        torch.save(legoModel, "legoModel.pk")
        print("Building sampler")
        sampler = LegoGramRNNSampler(legoModel)
        torch.save(sampler, "sampler.pk")
        print("Constracting dataset")
        dataset = MolecularNotationDataset(smiles,sampler,data)
        torch.save(dataset,'lg.bin')
    else:
        dataset = torch.load('lg.bin')

    train_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collect)
    device = torch.device('cpu')
    if args.cuda:
        device = torch.device('cuda')
    model = RNN(voc_size=dataset.vocsize, device=device)
    model.train()
    model.cuda()
    print(f"Model has been created on device {device}")
    smiles_dataset = dataset.smiles
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_f = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    writer = SummaryWriter(comment = args.name_task)
    losses = []
    out_counter = 0
    cnt = 0
    for epoch in range(args.num_epochs):
        loss_list =[]
        for iteration, (batch, lengths) in enumerate(tqdm(train_loader)):
            batch = batch.cuda()
            logits, endp_model = model(batch, lengths)
            print(logits.shape)
            print(batch.shape)
            loss = loss_f(logits[:, :, :-1], batch[:, 1:])

            loss_list.append(loss.item())
            writer.add_scalar("CrossEntropyLoss", loss_list[-1], iteration+epoch*len(train_loader))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % args.print_every == 0 and iteration > 0:
                model.eval()
                number_generate = 100

                res = model.sample(number_generate, dataset.model)
                writer.add_text("Molecules after generator", json.dumps([res]))
                valid = len(res) * 100 / number_generate
                print(res)
                print("valid : {} %".format(valid))
                writer.add_scalar("Valid", valid, cnt)
                res = [robust_standardizer(mol) for mol in res]
                res = list(filter(lambda x: x is not None, res))
                unique = len([elem for elem in res if elem not in smiles_dataset])

                writer.add_text("Unique mols", json.dumps([res]))
                print(f"There are unique mols {unique}")
                print(res)
                writer.add_scalar("Unique", unique, cnt)
                cnt += 1
                model.train()
        writer.flush()
        epoch_loss = np.mean(loss_list)
        print(f"Loss on epoch {epoch } is {epoch_loss}")
        if out_counter < args.stop_after and epoch>0:
            if losses[-1] <= epoch_loss:
                out_counter += 1
            else:
                out_counter = 0
                torch.save(model, "experiments/" + args.name_task + "/model.pt")
        if epoch == 0:
            torch.save(model, "experiments/" + args.name_task + "/model.pt")
        losses.append(epoch_loss)
    return losses


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train RNN generator')
    p.add_argument('--num_epochs', type=int, help="Number of epochs", default=20)
    p.add_argument('--batch_size', type=int, help="Batch size", default=512)
    p.add_argument('--cuda', help="Use GPU acceleration", action='store_true', default=False)
    p.add_argument("--train", help="Train or run", action = 'store_true', default = False)
    p.add_argument("--lr", type = float, help="Adam optimizer learning rate", default=0.001)
    p.add_argument("--create_dataset", help = "Create dataset or load from file", default = False, action='store_true')
    p.add_argument("--print_every", type = int, default= 100, help ="Print every n'th step")
    p.add_argument("--no-tensorboard", default = True, action='store_false', help = "No tensorboard logging")
    p.add_argument("--name_task", type = str, default ="default_name", help = "Name of task")
    p.add_argument("--stop_after", type = int, default = 3, help = "Early stopping")
    args = p.parse_args()
    if args.name_task == "default_name":
        args.name_task = str(f"bs_{args.batch_size}_lr_{args.lr}_ne_{args.num_epochs}")
    path = "experiments/"+args.name_task
    if not os.path.isdir(path):
        os.makedirs(path)
    if args.train:
        train(args)



