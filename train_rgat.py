#!/usr/bin/env python
# coding: utf-8
import os
import dgl
import tqdm
import math
import random
import numpy as np
import dgl.nn as dglnn
from dgl import apply_each

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from ogb.lsc import MAG240MDataset

import argparse

dataset = MAG240MDataset(root="/path/to/ogb/dataset")

num_features = dataset.num_paper_features
num_classes = dataset.num_classes

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# ----------------------------- Code Copy From BOT ----------------------------------- #
epsilon = 1 - math.log(2)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return sinusoid_table

def get_warmup_and_linear_decay(warmup_steps):
    
    warmup_steps = max(warmup_steps, 1)
    
    def fn(step):
        if step < warmup_steps:
            return 1
        elif step == warmup_steps:
            return 0.5
        else:
            pos_step = step - warmup_steps 
            return 0.5 * 0.75 ** (pos_step // 5)
    return fn

# ------------------------------------------------------------------------------------ #

class GATConv_BN(dglnn.GATConv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bn = nn.BatchNorm1d(self._num_heads * self._out_feats)
    
    def forward(self, graph, feat, get_attention=False):
        rst = super().forward(graph, feat, get_attention)
        
        # Here we first do the reshape then do the normalization
        if get_attention:
            rst, att = rst
        
        rst = rst.view(-1,self._num_heads * self._out_feats)
        rst = self.bn(rst)
        
        if get_attention:
            return rst, att
        else:
            return rst

def linear_init(linear_layer):
    negative_slope = math.sqrt(5)
    gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
    std = gain / math.sqrt(linear_layer.in_features)
    weight_bound = math.sqrt(3.0) * std
    nn.init.uniform_(linear_layer.weight, -weight_bound, weight_bound)

    if linear_layer.bias is not None:
        bias_bound = 1.0 / math.sqrt(linear_layer.in_features)
        nn.init.uniform_(linear_layer.bias, -bias_bound, bias_bound)
    return linear_layer


class RGAT(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, etypes, num_layers, num_heads, dropout,
                 pred_ntype):
        super().__init__()
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.skip_norms = nn.ModuleList()
        self.path_attns = nn.ModuleList()
        self.path_norms = nn.ModuleList()

        self.label_embed = nn.Embedding(num_classes, in_channels)
        self.m2v_fc = linear_init(nn.Linear(64, in_channels))
        
        self.convs.append(dglnn.HeteroGraphConv({
            etype: GATConv_BN(
                    in_channels, hidden_channels // num_heads, 
                    num_heads, allow_zero_in_degree=True) for etype in etypes}, aggregate='stack'
        ))
        self.skips.append(linear_init(nn.Linear(in_channels, hidden_channels)))
        self.skip_norms.append(nn.BatchNorm1d(hidden_channels))
        self.path_attns.append(linear_init(nn.Linear(hidden_channels, 1)))
        self.path_norms.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(dglnn.HeteroGraphConv({
                etype: GATConv_BN(
                        hidden_channels, hidden_channels // num_heads, 
                        num_heads, allow_zero_in_degree=True) for etype in etypes}, aggregate='stack'
            ))
            self.skips.append(linear_init(nn.Linear(hidden_channels, hidden_channels)))
            self.skip_norms.append(nn.BatchNorm1d(hidden_channels))
            self.path_attns.append(linear_init(nn.Linear(hidden_channels, 1)))
            self.path_norms.append(nn.BatchNorm1d(hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.label_mlp = nn.Sequential(
            nn.Linear(2*in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(0.3)

        self.hidden_channels = hidden_channels
        self.pred_ntype = pred_ntype
        self.etypes = etypes

    def forward(self, mfgs, x, m2v_x, label_y, label_idx):
        m2v_x = apply_each(m2v_x, self.m2v_fc)
        m2v_x = apply_each(m2v_x, self.input_drop)
        assert m2v_x.keys() == x.keys()
        x = dict([(k, x[k]+m2v_x[k]) for k in x.keys()])
        
        label_embed = self.label_embed(label_y)
        label_embed = self.input_drop(label_embed)
        label_feats = x['paper'][label_idx]
        label_embed = torch.cat([label_embed, label_feats], dim=1)
        label_embed = self.label_mlp(label_embed)
        x['paper'][label_idx] = label_embed

        for i in range(len(mfgs)):
            mfg = mfgs[i]
            x_cnt = [(k, mfg.num_dst_nodes(k)) for k in x.keys() if mfg.num_dst_nodes(k) > 0]
            x_cnt_c = np.cumsum([0,] + [a[1] for a in x_cnt])
            x_cnt = list(zip(x_cnt, x_cnt_c[:-1],x_cnt_c[1:]))

            x_dst = dict([(k, v[:mfg.num_dst_nodes(k)]) for k,v in x.items()])
            x_skip = apply_each(x_dst, self.skips[i])
            
            x_skip = torch.cat([x_skip[k[0][0]] for k in x_cnt], dim=0)
            x_skip = self.skip_norms[i](x_skip)
            x_skip = F.elu(x_skip)
            
            h = self.convs[i](mfg, (x, x_dst)) # B * P * HD
            h = torch.cat([h[k[0][0]] for k in x_cnt], dim=0)
            h = F.elu(h)
            
            x_skip = torch.cat([h, x_skip[:,None,:]], dim=1) # B * P+1 * HD
            x_attn = self.path_attns[i](x_skip) # B * P+1 * 1
            x_attn = F.softmax(x_attn, dim=1)
            x_skip = torch.matmul(x_attn.permute(0,2,1), x_skip).squeeze(1)
            x_skip = self.dropout(self.path_norms[i](x_skip))
            x = dict([(k[0], x_skip[b:e]) for k, b, e in x_cnt])
        return self.mlp(x['paper'])


def prepare_train(args, proc_id, devices):
    # ---------------------- Build Graph From Scratch ---------------------- #
    ei_writes = dataset.edge_index('author', 'writes', 'paper')
    ei_cites = dataset.edge_index('paper', 'paper')
    ei_affiliated = dataset.edge_index('author', 'institution')

    heter_graph = dgl.heterograph({
        ('author', 'write', 'paper'): (ei_writes[0], ei_writes[1]),
        ('paper', 'write-by', 'author'): (ei_writes[1], ei_writes[0]),
        ('author', 'affiliate-with', 'institution'): (ei_affiliated[0], ei_affiliated[1]),
        ('institution', 'affiliate', 'author'): (ei_affiliated[1], ei_affiliated[0]),
        ('paper', 'cite', 'paper'): (np.concatenate([ei_cites[0], ei_cites[1]]), np.concatenate([ei_cites[1], ei_cites[0]]))
        })

    heter_offset = {
        "author": [0, dataset.num_authors],
        "institution": [dataset.num_authors, dataset.num_authors+dataset.num_institutions],
        "paper":[dataset.num_authors+dataset.num_institutions, dataset.num_authors+dataset.num_institutions+dataset.num_papers]
    }
    # ---------------------- Load Features ---------------------- #
    part_feat = np.memmap(os.path.join(args.data_root, "full.npy"),
                          mode='r', dtype='float16', shape=(heter_graph.num_nodes(), num_features))
    # ---------------------- Load Year Info ---------------------- #
    part_year = np.memmap(os.path.join(args.data_root, "full_year.npy"),
                          mode='r', dtype='int', shape=(heter_graph.num_nodes(),))
    part_year = 2022 - torch.LongTensor(part_year.tolist())
    heter_graph.ndata['x_year'] = dict([(k, part_year[heter_offset[k][0]:heter_offset[k][1]]) for k in heter_graph.ntypes])
    # ---------------------- Load metapaht2vec Info ---------------------- #
    part_m2v = np.memmap(os.path.join(args.data_root, "baidu_m2v.npy"),
                         mode='r', dtype='float16', shape=(heter_graph.num_nodes(), 64))
    # ----------------------  Load labels  ---------------------- #
    label_refer = torch.LongTensor(np.nan_to_num(dataset.paper_label.tolist(), copy=False, nan=-1))
    label_dict = dict()
    for key in heter_graph.ntypes:
        if key == 'paper':
            label = label_refer
        else:
            label = torch.ones(heter_graph.num_nodes(key)).long() * -1
        label_dict[key] = label
    heter_graph.ndata['y'] = label_dict
    # --------------------  Load train / Valid index  ------------------- #
    train_org_idx = dataset.get_idx_split('train')
    if args.cv_name is None:
        train_idx = train_org_idx
        valid_idx = dataset.get_idx_split('valid')
    else:
        train_aux_list = [x for x in os.listdir(args.cv_root) if x != args.cv_name]
        train_aux_idx = np.concatenate([np.load(os.path.join(args.cv_root, x)) for x in train_aux_list])
        train_idx = np.concatenate([train_org_idx, train_aux_idx])
        valid_idx = np.load(os.path.join(args.cv_root, args.cv_name))

    train_idx = {'paper':torch.LongTensor(train_idx)}
    proc_size = np.ceil(len(valid_idx) / len(devices))
    begin_id = int(proc_id * proc_size)
    end_id = int(min(len(valid_idx), (proc_id+1) * proc_size))
    valid_idx = {'paper':torch.LongTensor(valid_idx[begin_id:end_id])}

    # --------------------  Wrap everything into dataloader  ------------------- #
    
    train_sampler = dgl.dataloading.MultiLayerNeighborSampler([
        {'write-by':10,'affiliate':5,'cite':15,'write':10,'affiliate-with':0},
        {'write-by':0,'write':10,'affiliate-with':0,'affiliate':0,'cite':25},
    ])

    train_dataloader = dgl.dataloading.DataLoader(
        heter_graph, train_idx, train_sampler, 
        batch_size=args.train_batch_size,
        shuffle=True, drop_last=False, num_workers=0,
        device='cuda:{}'.format(devices[proc_id]), use_ddp=len(args.gpus.split(','))>1
    )
    # Here we need to define a different sampler for validation process
    # Check the code: https://github.com/PaddlePaddle/PGL/blob/7384ec4eb36a2ec94fd9fabd407a6150e50f3cda/examples/kddcup2021/MAG240M/r_unimp/dataset/data_generator_r_unimp_multi_gpu_infer.py#L344 
    # And this as well: https://github.com/PaddlePaddle/PGL/blob/7384ec4eb36a2ec94fd9fabd407a6150e50f3cda/examples/kddcup2021/MAG240M/r_unimp/r_unimp_multi_gpu_infer.py#L68
    valid_dataloader = dgl.dataloading.DataLoader(
        heter_graph, valid_idx, train_sampler, 
        batch_size=args.valid_batch_size,
        shuffle=False, drop_last=False, num_workers=0,
        device='cuda:{}'.format(devices[proc_id]), use_ddp=False
    )
    return part_feat, part_m2v, train_dataloader, valid_dataloader


def train(proc_id, args, devices):
    dev_id = devices[proc_id]
    if len(devices) > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port=args.port)
        world_size = len(devices)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=proc_id)

    torch.cuda.set_device(dev_id)

    YEAR_EMBED = torch.FloatTensor(get_sinusoid_encoding_table(200, 768)).to(dev_id)
    # LABEL_EMBED = torch.cat([torch.zeros(1,num_classes),torch.eye(num_classes)]).to(dev_id)
    
    part_feat, part_m2v, train_dataloader, valid_dataloader = prepare_train(args, proc_id, devices)

    print('Initializing model...')
    model = RGAT(num_features, num_classes, 
                 1024, ['affiliate-with', 'write', 'affiliate', 'cite', 'write-by'], 
                 2, 4, 0.5, 'paper').to(dev_id)

    # convert BN to SyncBatchNorm. see https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
    if len(devices) > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id, find_unused_parameters=True)

    # refer: https://github.com/PaddlePaddle/PGL/blob/7384ec4eb36a2ec94fd9fabd407a6150e50f3cda/examples/kddcup2021/MAG240M/r_unimp/optimization.py#L36
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, get_warmup_and_linear_decay(25))

    train_idx = train_dataloader.indices['paper']
    train_bool = torch.scatter(torch.zeros(dataset.num_papers), 0, train_idx, torch.ones(len(train_idx))).to(dev_id)

    best_acc = 0

    for i in range(args.epochs):
        # make shuffling work properly across multiple epochs.
        # see https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            
            for j, (_, _, mfgs) in enumerate(tq):
                x, m2v_x = dict(), dict()
                for key in ['author','institution','paper']:
                    if key == 'author':
                        feat_idx = (mfgs[0].srcdata['_ID'][key]).cpu()
                    elif key == 'institution':
                        feat_idx = (mfgs[0].srcdata['_ID'][key] + dataset.num_authors).cpu()
                    else:
                        feat_idx = (mfgs[0].srcdata['_ID'][key] + dataset.num_authors + dataset.num_institutions).cpu()
                    text_feat = torch.FloatTensor(part_feat[feat_idx]).to(dev_id)
                    year_feat = YEAR_EMBED[mfgs[0].srcdata['x_year'][key]]
                    x[key] = text_feat + year_feat
                    m2v_feat = torch.FloatTensor(part_m2v[feat_idx]).to(dev_id)
                    m2v_x[key] = m2v_feat

                label_index = torch.arange(mfgs[0].num_src_nodes('paper')).to(dev_id)
                label_index = label_index[train_bool[mfgs[0].srcdata['_ID']['paper']]==1]
                label_index = label_index[label_index >= mfgs[-1].num_dst_nodes('paper')]
                label_y = mfgs[0].srcdata['y']['paper'][label_index]

                rd_y = torch.randint(0, num_classes, (len(label_index),)).to(dev_id)
                rd_m = (torch.rand(len(label_index)) < 0.15).to(dev_id)
                label_y[rd_m] = rd_y[rd_m]

                y = mfgs[-1].dstdata['y']['paper']
                y_hat = model(mfgs, x, m2v_x, label_y, label_index)
                
                loss = F.cross_entropy(y_hat, y)

                opt.zero_grad()
                loss.backward()
                opt.step()
                acc = (y_hat.argmax(1) == y).float().mean()
                tq.set_postfix({'loss': '%.4f' % loss.item(), 'acc': '%.4f' % acc.item()}, refresh=False)

        # eval in each process
        model.eval()
        correct = torch.LongTensor([0]).to(dev_id)
        total = torch.LongTensor([0]).to(dev_id)
        for j, (_, _, mfgs) in enumerate(tqdm.tqdm(valid_dataloader)):
            with torch.no_grad():
                x, m2v_x = dict(), dict()
                for key in ['author','institution','paper']:
                    if key == 'author':
                        feat_idx = (mfgs[0].srcdata['_ID'][key]).cpu()
                    elif key == 'institution':
                        feat_idx = (mfgs[0].srcdata['_ID'][key] + dataset.num_authors).cpu()
                    else:
                        feat_idx = (mfgs[0].srcdata['_ID'][key] + dataset.num_authors + dataset.num_institutions).cpu()
                    text_feat = torch.FloatTensor(part_feat[feat_idx]).to(dev_id)
                    year_feat = YEAR_EMBED[mfgs[0].srcdata['x_year'][key]]
                    x[key] = text_feat + year_feat
                    m2v_feat = torch.FloatTensor(part_m2v[feat_idx]).to(dev_id)
                    m2v_x[key] = m2v_feat

                label_index = torch.arange(mfgs[0].num_src_nodes('paper')).to(dev_id)
                label_index = label_index[train_bool[mfgs[0].srcdata['_ID']['paper']]==1]
                label_index = label_index[label_index >= mfgs[-1].num_dst_nodes('paper')]
                label_y = mfgs[0].srcdata['y']['paper'][label_index]

                y = mfgs[-1].dstdata['y']['paper']
                y_hat = model(mfgs, x, m2v_x, label_y, label_index)
                
                correct += (y_hat.argmax(1) == y).sum().item()
                total += y_hat.shape[0]

        # `reduce` data into process 0
        if len(devices) > 1:
            torch.distributed.reduce(correct, dst=0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(total, dst=0, op=torch.distributed.ReduceOp.SUM)
        
        acc = (correct / total).item()

        sched.step()

        # process 0 print accuracy and save model
        if proc_id == 0:
            print('Epoch {} Validation accuracy:'.format(i), acc)

            if best_acc < acc:
                best_acc = acc
                print('Updating best model...')
                torch.save(model.state_dict(), args.model_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./part', help="root directory for partitioned data.")
    parser.add_argument('--train_batch_size', type=int, default=1024, help="batch size for training.")
    parser.add_argument('--valid_batch_size', type=int, default=128, help="batch size for validation.")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--model-path', type=str, default='./model_ddp.pt', help='Path to store the best model.')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help="target gpus")
    parser.add_argument('--cv_root', type=str, default='./valid_split/', help="save root for you cross validation files")
    parser.add_argument('--cv_name', type=str, default=None, help="cross validation file name")
    parser.add_argument('--port', type=str, default='12345', help="sever running port")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    args = parser.parse_args()

    devices = list(map(int, args.gpus.split(',')))
    n_gpus = len(devices)

    set_seed(args.seed)

    if n_gpus <= 1:
        train(0, args, devices)
        
    else:
        mp.spawn(train, args=(args, devices), nprocs=n_gpus)
