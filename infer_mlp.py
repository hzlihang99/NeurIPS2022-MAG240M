import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d, Identity

from ogb.lsc import MAG240MDataset, MAG240MEvaluator

class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = True, relu_last: bool = False):
        super(MLP, self).__init__()

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear((num_layers -1)* hidden_channels, out_channels))

        self.batch_norms = ModuleList()
        for _ in range(num_layers - 1):
            norm = BatchNorm1d(hidden_channels) if batch_norm else Identity()
            self.batch_norms.append(norm)

        self.dropout = dropout
        self.relu_last = relu_last

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, x):
        xs = []
        for lin, batch_norm in zip(self.lins[:-1], self.batch_norms):
            x = lin(x)
            if self.relu_last:
                x = batch_norm(x).relu_()
            else:
                x = batch_norm(x.relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = torch.cat(xs, 1)
        x = self.lins[-1](x)
        return x

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return sinusoid_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=3),
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--relu_last', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--batch_size', type=int, default=380000)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--version', type=int, default=0)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    ROOT = '/your/ogb/lsc/dataset'
    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()
    author_offset = 0
    inst_offset = author_offset + dataset.num_authors
    paper_offset = inst_offset + dataset.num_institutions
    logits_file = './logit_{}.npy'.format(args.fold)
    test_idx = dataset.get_idx_split('test-challenge')
    
    t = time.perf_counter()
    features = np.memmap('./full_diffusion.npy', mode='r', dtype='float16', shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers, dataset.num_paper_features))
    logits_features = np.memmap(logits_file, mode='r', dtype='float32', shape=(dataset.num_papers, dataset.num_classes))
    year_features = np.memmap('./year_diffusion.npy', mode='r', dtype='float16', shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers, 121))
    metapath = np.memmap('./m2v_embed.npy', mode='r', dtype='float16', shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers, 64))
    year_embed = torch.FloatTensor(get_sinusoid_encoding_table(200, 768))
    year_embed = torch.nn.Embedding.from_pretrained(year_embed, freeze=True)
    years = torch.from_numpy(dataset.paper_year)
    years = years - years.min()

    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading test node features...', end=' ', flush=True)
    x_test = torch.from_numpy(features[test_idx + paper_offset])
    x_logits_test = torch.from_numpy(logits_features[test_idx])
    x_year_test = torch.from_numpy(year_features[test_idx + paper_offset])
    x_metapath_test = torch.from_numpy(metapath[test_idx])
    year_test = years[test_idx]
    x_test = x_test + year_embed(year_test)

    x_test = torch.cat((x_test, x_logits_test, x_year_test, x_metapath_test), dim=1)
    x_test = x_test.to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    model = MLP(dataset.num_paper_features+dataset.num_classes+121+64, args.hidden_channels,
                dataset.num_classes, args.num_layers, args.dropout,
                not args.no_batch_norm, args.relu_last).to(device)
    
    model.load_state_dict(torch.load('mlp_fold{}.pt'.format(args.fold)))
    model = model.to(device)

    with torch.no_grad():
       model.eval()
       res = {'y_pred': model(x_test).argmax(dim=-1)}
       evaluator.save_test_submission(res, 'results/mlp', mode = 'test-dev')