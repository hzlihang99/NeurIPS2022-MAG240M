import os
import tqdm
import torch
import argparse
import numpy as np
from ogb.lsc import MAG240MDataset

import dgl
import dgl.function as fn

parser = argparse.ArgumentParser()
parser.add_argument("--rootdir", type=str, default=".", help="Directory to download the OGB dataset.",)
parser.add_argument("--m2vdir", type=str, default=".", help="Directory saved pre-trained meta2path representation.",)
parser.add_argument("--savedir", type=str, default=".", help="Directory to store preprocessed features.")
args = parser.parse_args()

print("Building graph")
dataset = MAG240MDataset(root=args.rootdir)
ei_writes = dataset.edge_index("author", "writes", "paper")
ei_cites = dataset.edge_index("paper", "paper")
ei_affiliated = dataset.edge_index("author", "institution")

# We sort the nodes starting with the papers, then the authors, then the institutions.
author_offset = 0
inst_offset = author_offset + dataset.num_authors
paper_offset = inst_offset + dataset.num_institutions

g = dgl.heterograph(
    {
        ("author", "write", "paper"): (ei_writes[0], ei_writes[1]),
        ("paper", "write-by", "author"): (ei_writes[1], ei_writes[0]),
        ("author", "affiliate-with", "institution"): (
            ei_affiliated[0],
            ei_affiliated[1],
        ),
        ("institution", "affiliate", "author"): (
            ei_affiliated[1],
            ei_affiliated[0],
        ),
        ("paper", "cite", "paper"): (
            np.concatenate([ei_cites[0], ei_cites[1]]),
            np.concatenate([ei_cites[1], ei_cites[0]]),
        ),
    }
)

paper_feat = dataset.paper_feat
author_feat = np.memmap(
    os.path.join(args.savedir, "author_feat.npy"), mode="w+", dtype="float16", 
    shape=(dataset.num_authors, dataset.num_paper_features),
)
inst_feat = np.memmap(
    os.path.join(args.savedir, "inst_feat.npy"), mode="w+", dtype="float16", 
    shape=(dataset.num_institutions, dataset.num_paper_features),
)

# Iteratively process author features along the feature dimension.
BLOCK_COLS = 16
with tqdm.trange(0, dataset.num_paper_features, BLOCK_COLS) as tq:
    for start in tq:
        tq.set_postfix_str("Reading paper features...")
        g.nodes["paper"].data["x"] = torch.FloatTensor(
            paper_feat[:, start : start + BLOCK_COLS].astype("float32")
        )
        # Compute author features...
        tq.set_postfix_str("Computing author features...")
        g.update_all(fn.copy_u("x", "m"), fn.mean("m", "x"), etype="write-by")
        # Then institution features...
        tq.set_postfix_str("Computing institution features...")
        g.update_all(
            fn.copy_u("x", "m"), fn.mean("m", "x"), etype="affiliate-with"
        )
        tq.set_postfix_str("Writing author features...")
        author_feat[:, start : start + BLOCK_COLS] = (
            g.nodes["author"].data["x"].numpy().astype("float16")
        )
        tq.set_postfix_str("Writing institution features...")
        inst_feat[:, start : start + BLOCK_COLS] = (
            g.nodes["institution"].data["x"].numpy().astype("float16")
        )
        del g.nodes["paper"].data["x"]
        del g.nodes["author"].data["x"]
        del g.nodes["institution"].data["x"]

author_feat.flush()
inst_feat.flush()

# Process feature
full_feat = np.memmap(
    os.path.join(args.savedir, "full_feat.npy"), mode="w+", dtype="float16",
    shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers, dataset.num_paper_features,),
)
BLOCK_ROWS = 100000
for start in tqdm.trange(0, dataset.num_authors, BLOCK_ROWS):
    end = min(dataset.num_authors, start + BLOCK_ROWS)
    full_feat[author_offset + start : author_offset + end] = author_feat[start:end]
for start in tqdm.trange(0, dataset.num_institutions, BLOCK_ROWS):
    end = min(dataset.num_institutions, start + BLOCK_ROWS)
    full_feat[inst_offset + start : inst_offset + end] = inst_feat[start:end]
for start in tqdm.trange(0, dataset.num_papers, BLOCK_ROWS):
    end = min(dataset.num_papers, start + BLOCK_ROWS)
    full_feat[paper_offset + start : paper_offset + end] = paper_feat[start:end]

# Prepare the year feature

paper_year = dataset.paper_year
author_year = np.memmap(
    os.path.join(args.savedir, 'author_year.npy'), mode='w+', 
    dtype='int', shape=(dataset.num_authors,)
)
inst_year = np.memmap(
    os.path.join(args.savedir, "inst_year.npy"), mode='w+', 
    dtype='int', shape=(dataset.num_institutions,)
)

g.nodes['paper'].data['x'] = torch.FloatTensor(paper_year.astype('int'))
g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='write-by')
g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='affiliate-with')
author_year[:,] = g.nodes['author'].data['x'].numpy().astype('int')
inst_year[:,] = g.nodes['institution'].data['x'].numpy().astype('int')

del g.nodes['paper'].data['x']
del g.nodes['author'].data['x']
del g.nodes['institution'].data['x']

author_year.flush()
inst_year.flush()

full_year = np.memmap(
        os.path.join(args.savedir, "full_year.npy"), mode='w+', dtype='int',
        shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers,)
)

BLOCK_ROWS = 100000
for start in tqdm.trange(0, dataset.num_authors, BLOCK_ROWS):
    end = min(dataset.num_authors, start + BLOCK_ROWS)
    full_year[author_offset + start:author_offset + end] = author_year[start:end]
for start in tqdm.trange(0, dataset.num_institutions, BLOCK_ROWS):
    end = min(dataset.num_institutions, start + BLOCK_ROWS)
    full_year[inst_offset + start:inst_offset + end] = inst_year[start:end]
for start in tqdm.trange(0, dataset.num_papers, BLOCK_ROWS):
    end = min(dataset.num_papers, start + BLOCK_ROWS)
    full_year[paper_offset + start:paper_offset + end] = paper_year[start:end]


part_num = (dataset.num_papers + dataset.num_authors + dataset.num_institutions + 1) // 10
start_idx = 0
node_chunk_size = 100000
m2v_merge_feat = np.memmap(
    os.path.join(args.savedir, 'm2v_embed.npy'), dtype=np.float16, mode='w+', 
    shape=(dataset.num_papers + dataset.num_authors + dataset.num_institutions, 64)
)
files = os.listdir(args.m2vdir)
files = sorted(files)
for idx, start_idx in enumerate(range(0, dataset.num_papers + dataset.num_authors + dataset.num_institutions, part_num)):
    end_idx = min(dataset.num_papers + dataset.num_authors + dataset.num_institutions, start_idx + part_num)
    f = os.path.join(args.m2vdir, files[idx])
    m2v_feat_tmp = np.memmap(f, dtype=np.float16, mode='r', shape=(end_idx - start_idx, 64))
    for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
        j = min(i + node_chunk_size, end_idx)
        m2v_merge_feat[i: j] = m2v_feat_tmp[i - start_idx: j - start_idx]
    m2v_merge_feat.flush()
    del m2v_feat_tmp


g = dgl.to_homogeneous(g)
g.edata["etype"] = g.edata[dgl.ETYPE].byte()
g = g.formats("csc")
dgl.save_graphs(os.path.join(args.savedir, 'homo_graph.bin'), g)