import time
import sys

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
# from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer= SummaryWriter()

import sys
sys.path.append(".")

import os.path as osp
import uuid
import pickle
import datetime
import random
import torch
import os

import argparse

from torch.utils.tensorboard import SummaryWriter
writer= SummaryWriter()

import numpy as np

import torch.optim as optim

from torch.optim.lr_scheduler import MultiStepLR

torch.backends.cudnn.deterministic = True
# torch.manual_seed(12345)
# torch.cuda.manual_seed_all(12345)
# random.seed(12345)
# np.random.seed(12345)

from gnn.dd_model import PershomLearnedFilt, PershomLearnedFiltSup, PershomRigidDegreeFilt, GIN, SimpleNNBaseline, ClassicGNN, ClassicReadoutFilt
from data.data import dataset_factory, train_test_val_split, Subset
from data.utils import my_collate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        assert isinstance(indices, (list, tuple))
        self.ds = dataset
        self.indices = tuple(indices)
        self.y = dataset.y[indices]
        assert len(indices) <= len(dataset)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.ds[self.indices[idx]]


def cross_validation_with_val_set(args, dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None):

    val_losses, accs, durations = [], [], []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds, args.device))):
        # print("train_idx: ", train_idx)
        tr_dataset = Subset(dataset, train_idx)
        te_dataset = Subset(dataset, test_idx)
        val_dataset = Subset(dataset, val_idx)
        dl_train = torch.utils.data.DataLoader(
            tr_dataset,
            collate_fn=my_collate,
            batch_size=args.batch_size,  # make this BIG for better contrastive learning
            shuffle=True,
            # if last batch would have size 1 we have to drop it ...
            drop_last=(len(tr_dataset) % args.batch_size == 1)
        )

        dl_val = torch.utils.data.DataLoader(
            val_dataset,
            collate_fn=my_collate,
            batch_size=1,
            shuffle=False,
            # drop_last = (len(dataset) % 64 == 1)
        )
        dl_test = torch.utils.data.DataLoader(
            te_dataset,
            collate_fn=my_collate,
            batch_size=1,
            shuffle=False,
            # drop_last = (len(dataset) % 64 == 1)
        )
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train_ph(args, model, optimizer, dl_train)
            val_losses.append(eval_loss(model, dl_val))
            accs.append(eval_acc(model, dl_test))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }
            print(eval_info, flush= True)

            writer.add_scalars('loss_' + args.exp_name, {'train_loss_fold'+str(fold): float(train_loss)}, epoch)
            writer.add_scalars('acc_' + args.exp_name, {'test_acc_fold'+str(fold): float(accs[-1])}, epoch)
            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
        fold_i= fold+1
        loss, acc = loss.view(fold_i, epochs), acc.view(fold_i, epochs)
        loss, argmin = loss.min(dim=1)
        acc = acc[torch.arange(fold_i, dtype=torch.long), argmin]

        loss_mean = loss.mean().item()
        acc_mean = acc.mean().item()
        acc_std = acc.std().item()
        duration_mean = duration.mean().item()
        print('AFTER FOLD '+str(fold)+f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
              f'± {acc_std:.3f}, Duration: {duration_mean:.3f}', flush= True)
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Duration: {duration_mean:.3f}', flush= True)

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, folds, device):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    y= torch.tensor([dataset[i].y for i in range(len(dataset))], dtype= torch.long)
    for _, idx in skf.split(torch.zeros(len(dataset)), y):
        test_indices.append(list(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        # train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
        train_indices.append(list(train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()))
    return (list(train_indices)), (list(test_indices)), (list(val_indices))

def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)

def train_ph(args, model, opt, dl_train):
    model.train()

    total_loss = 0

    for batch_i, batch in enumerate(dl_train, start=1):
        opt.zero_grad()
        batch = batch.to(device)
        if not hasattr(batch, 'node_lab'): batch.node_lab = None
        batch.boundary_info = [e.to(device) for e in batch.boundary_info]
        logit= (model(batch,device)).to(device)
        loss1 = F.nll_loss(logit.float(), batch.y.view(-1).long())
        loss= loss1
        loss.backward()
        total_loss += loss.item() * num_graphs(batch)
        opt.step()
    return total_loss / len(dl_train.dataset)

def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data, device)
        loss = F.nll_loss(out.float(), data.y.view(-1).long())
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data, device).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data, device)
        loss += F.nll_loss(out.float(), data.y.view(-1).long(), reduction='sum').item()
    return loss / len(loader.dataset)

if __name__ == "__main__":
    print("DD TRAIN")

    parser = argparse.ArgumentParser(description='DD')

    parser.add_argument('--device', type= int, default= 0)

    parser.add_argument('--max_process_on_device', type=int)
    parser.add_argument('--readout', type=str, default="extph")
    parser.add_argument('--exp_name', type= str, default= 'dd', help= 'experiment name to save as')
    parser.add_argument('--dataset_name', type= str, default= 'DD')
    # parser.add_argument('--selfsupervised', dest='ssl', default= True, type= bool)#action='store_false')
    # parser.add_argument('--evaluation', type=str, default= 'SVC', choices= ['RandomForest', 'LogisticRegression', 'SVC'], help= 'downstream evaluation protocol classifier type')
    parser.add_argument('--seed', dest= 'seed', default= 0)
    parser.add_argument('--verbose', type= bool, default= True)
    parser.add_argument('--sup_combo', dest= 'sup', default= True, type= bool)
    parser.add_argument('--lr', type= float, default= 0.01)
    parser.add_argument('--lr_drop_fact', type= float, default= 0.5)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--epoch_step', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)#512)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--use_super_level_set_filtration', type=bool, default=True)
    parser.add_argument('--use_raw_node_label', type= bool, default= True)
    parser.add_argument('--use_node_degree', type=bool, default=True)
    parser.add_argument('--set_node_degree_uninformative', type=bool, default=False)
    parser.add_argument('--use_node_label', type=bool, default=True)
    parser.add_argument('--filt_conv_number', type=int, default=1)
    parser.add_argument('--filt_conv_dimension', type=int, default=128)
    parser.add_argument('--conv_number', type=int, default=5)
    parser.add_argument('--conv_dimension', type=int, default=128)
    parser.add_argument('--gin_mlp_type', type=str, default='lin_bn_lrelu_lin')
    parser.add_argument('--num_struct_elements', type=int, default=128)
    parser.add_argument('--cls_hidden_dimension', type=int, default=512)
    parser.add_argument('--drop_out', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default= 'results')
    parser.set_defaults(ssl=False)

    args = parser.parse_args()
    print(args)

    device= args.device

    dataset = dataset_factory(args.dataset_name, verbose=args.verbose)

    if args.readout == "extph":
        model = PershomLearnedFiltSup(dataset, args.use_super_level_set_filtration, args.use_node_degree,
                                           args.set_node_degree_uninformative, args.use_node_label,
                                           args.use_raw_node_label,
                                           args.filt_conv_number, args.filt_conv_dimension, args.gin_mlp_type,
                                           args.num_struct_elements, args.cls_hidden_dimension, args.drop_out,
                                           conv_number=args.conv_number, conv_dimension=args.conv_dimension, aug=None,
                                           readout=args.readout).to(device)
    else:
        model = ClassicReadoutFilt(dataset, args.use_super_level_set_filtration, args.use_node_degree,
                                   args.set_node_degree_uninformative, args.use_node_label,
                                   args.use_raw_node_label,
                                   args.filt_conv_number, args.filt_conv_dimension, args.gin_mlp_type,
                                   args.num_struct_elements, args.cls_hidden_dimension, args.drop_out,
                                   conv_number=args.conv_number, conv_dimension=args.conv_dimension, aug=None,
                                   readout=args.readout).to(device)

    cross_validation_with_val_set(args, dataset, model, 10, args.num_epochs, args.batch_size,
                                  args.lr, lr_decay_factor=0.5, lr_decay_step_size= 50,
                                  weight_decay=0)
