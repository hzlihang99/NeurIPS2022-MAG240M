import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


def train(model, x_train, y_train, batch_size, optimizer, weight=None):
    model.train()

    total_loss = 0
    for idx in DataLoader(range(y_train.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        # import ipdb; ipdb.set_trace()
        if weight is None:
            w = torch.ones(idx.shape[0]).to(y_train)
        else:
            w = weight[idx]
        loss = F.cross_entropy(model(x_train[idx]), y_train[idx], reduction='none')
        loss = (loss * w).mean()
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()

    return total_loss / y_train.size(0)


@torch.no_grad()
def test(model, x_eval, y_eval, evaluator):
    model.eval()
    y_pred = model(x_eval).argmax(dim=-1)
    return evaluator.eval({'y_true': y_eval, 'y_pred': y_pred})['acc']

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
    valid_file = './valid_split/valid_{}.npy'.format(args.fold)
    valid_idx = np.load(valid_file)
    train_idx = dataset.get_idx_split('train')
    all_valid_idx = dataset.get_idx_split('valid')
    new_train_idx = np.setdiff1d(all_valid_idx, valid_idx)
    train_idx = np.concatenate((train_idx, new_train_idx), axis=0)
    test_idx = dataset.get_idx_split('test-dev')
    t = time.perf_counter()
    features = np.memmap('./full_diffusion.npy', mode='r', dtype='float16', shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers, dataset.num_paper_features))
    logits_features = np.memmap(logits_file, mode='r', dtype='float32', shape=(dataset.num_papers, dataset.num_classes))
    year_features = np.memmap('./year_diffusion.npy', mode='r', dtype='float16', shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers, 121))
    metapath = np.memmap('./m2v_embed.npy', mode='r', dtype='float16', shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers, 64))
    year_embed = torch.FloatTensor(get_sinusoid_encoding_table(200, 768))
    year_embed = torch.nn.Embedding.from_pretrained(year_embed, freeze=True)
    years = torch.from_numpy(dataset.paper_year)
    years = years - years.min()

    print('Reading training node features...', end=' ', flush=True)
    x_train = torch.from_numpy(features[train_idx + paper_offset])
    x_logits_train = torch.from_numpy(logits_features[train_idx])
    print('Reading feature done')
    x_year_train = torch.from_numpy(year_features[train_idx + paper_offset])
    x_metapath_train = torch.from_numpy(metapath[train_idx])
    print('year, metapath done')
    year_train = years[train_idx]
    x_train = x_train + year_embed(year_train)
    x_train = torch.cat((x_train, x_logits_train, x_year_train, x_metapath_train), dim=1)
    x_train = x_train.to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading validation node features...', end=' ', flush=True)
    x_valid = torch.from_numpy(features[valid_idx + paper_offset])
    x_logits_valid = torch.from_numpy(logits_features[valid_idx])
    x_year_valid = torch.from_numpy(year_features[valid_idx + paper_offset])
    x_metapath_valid = torch.from_numpy(metapath[valid_idx])
    year_valid = years[valid_idx]
    x_valid = x_valid + year_embed(year_valid)

    x_valid = torch.cat((x_valid, x_logits_valid, x_year_valid, x_metapath_valid), dim=1)
    x_valid = x_valid.to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading test node features...', end=' ', flush=True)
    x_test = features[test_idx + paper_offset]
    x_test = torch.from_numpy(x_test).to(torch.float).to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    y_train = torch.from_numpy(dataset.paper_label[train_idx])
    y_train = y_train.to(device, torch.long)
    y_valid = torch.from_numpy(dataset.paper_label[valid_idx])
    y_valid = y_valid.to(device, torch.long)
    
    model = MLP(dataset.num_paper_features+dataset.num_classes+121+64, args.hidden_channels,
                dataset.num_classes, args.num_layers, args.dropout,
                not args.no_batch_norm, args.relu_last).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f'#Params: {num_params}')

    best_valid_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, x_train, y_train, args.batch_size, optimizer, weight=None)
        train_acc = test(model, x_train, y_train, evaluator)
        valid_acc = test(model, x_valid, y_valid, evaluator)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            print('saving model')
            if epoch > 100:
                torch.save(model.state_dict(), './mlp_fold_{}_{}.pt'.format(args.fold, args.version))
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
                  f'Best: {best_valid_acc:.4f}')
