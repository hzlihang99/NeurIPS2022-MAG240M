
import os
import dgl
import tqdm
import torch
import argparse
import numpy as np
import dgl.function as fn
from ogb.lsc import MAG240MDataset

import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./part', help="root directory for partitioned data.")
    parser.add_argument('--feat_path', type=str, default='./full_feat.npy', help="path to your full feature matrix")
    parser.add_argument('--save_root', type=str, default='./', help="path to your full feature matrix")
    args = parser.parse_args()

    dataset = MAG240MDataset(root=args.data_root)
    author_offset = 0
    inst_offset = author_offset + dataset.num_authors
    paper_offset = inst_offset + dataset.num_institutions

    years = torch.from_numpy(dataset.paper_year)
    years = years - years.min()
    one_hot_year = F.one_hot(years)

    year_mem = np.memmap(
        './one_hot_years.npy', mode="w+", dtype="float16",
        shape=(dataset.num_authors+dataset.num_institutions+dataset.num_papers, one_hot_year.shape[1]),
    )
    year_mem[paper_offset:] = one_hot_year


    (g,), _ = dgl.load_graphs('./homo_graph.bin')

    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5)
    norm = norm.unsqueeze(1)

    features = np.memmap(args.feat_path, mode='r', dtype='float16', shape=(g.num_nodes(), dataset.num_paper_features))

    full_diffusion_feat = np.memmap(
        os.path.join(args.save_root, 'full_diffusion.npy'), mode="w+", 
        dtype="float16", shape=(g.num_nodes(), dataset.num_paper_features),
    )

    BLOCK_COLS = 16
    alpha = 0.05
    with tqdm.trange(0, dataset.num_paper_features, BLOCK_COLS) as tq:
        for start in tq:
            tq.set_postfix_str('Reading features')
            feat = torch.FloatTensor(features[:, start: start+BLOCK_COLS].astype('float16'))
            feat_0 = feat
            for _ in range(5):
                feat = feat * norm
                g.ndata['h'] = feat
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feat = g.ndata.pop('h')
                feat = feat * norm
                feat = (1-alpha)*feat + alpha*feat_0
            tq.set_postfix_str("Writing features...")
            full_diffusion_feat[:, start:start+BLOCK_COLS] = feat.numpy().astype('float16')
    full_diffusion_feat.flush()
    
    
    # Year Info Diffusion #
    year_diffusion_feat = np.memmap(
        os.path.join(args.save_root, 'year_diffusion.npy'), mode="w+", 
        dtype="float16", shape=(g.num_nodes(), 121),
    )

    BLOCK_COLS = 16
    alpha = 0.2
    with tqdm.trange(0, 121, BLOCK_COLS) as tq:
        for start in tq:
            tq.set_postfix_str('Reading features')
            feat = torch.FloatTensor(year_mem[:, start: start+BLOCK_COLS].astype('float16'))
            feat_0 = feat
            for _ in range(5):
                feat = feat * norm
                g.ndata['h'] = feat
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feat = g.ndata.pop('h')
                feat = feat * norm
                feat = (1-alpha)*feat + alpha*feat_0
            tq.set_postfix_str("Writing features...")
            year_diffusion_feat[:, start:start+BLOCK_COLS] = feat.numpy().astype('float16')
    year_diffusion_feat.flush()

