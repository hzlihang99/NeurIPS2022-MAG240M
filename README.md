# Source Code for NeurIPS2022 MAG240M Competition

## Step 0: Package Requirements

```
tqdm
ogb==1.3.4
dgl==0.9.0
torch==1.12.1
```
## Step 1: Download Auxiliary Data
Since our algorihtm use the KDD2021 winning teams' R-GAT as our base model, here we reused their pre-trained meta2path representation. Corresponding files can be downloaded: [here (password: 0mr0)](https://pan.baidu.com/s/1_0PhbFglsWmYdo9fO1CRGQ)

## Step 2: Preprocess Features

```
python prep_feat.py --rootdir /your/path/to/ogb/dataset --m2vdir /your/path/to/save/download_m2v/ --savedir ./
```

This will give you the following files:

* `author_feat.npy`: The author features, preprocessed by averaging the neighboring paper features.
* `inst_feat.npy`: The institution features, preprocessed by averaging the neighboring author features.
* `full_feat.npy`: The concatenated author, institution, and paper features.
* `author_year.npy`: The author year, preprocessed by averaging the neighboring paper years.
* `inst_year.npy` The institution years, preprocessed by averaging the neighboring author years.
* `full_year.npy`: The concatenated author, institution, and paper years.
* `m2v_embed.npy`: The concatenated author, institution, and paper m2v embed.


## Step 3: Generate 5-Fold Validation Splits

```
python split_valid.py
```

Then, you will save the new cross validation data in follow dir:
* `./valid_split`

## Step 4: Train the backbone R-GAT Model

```
# Fold k training:
python train_rgat.py --data_root ./ --cv_root valid_split/ --cv_name valid_k.npy --model-path ./rgat_k.pt
```

## Step 5: Infer the logits with trained R-GAT Model
```
# Fold k Inference:
python infer_rgat.py --data_root ./ --model-path ./rgat_k.pt --save-path rgat_k.npy --cv_root valid_split/ --cv_name logit_k.npy
```

## Step 6: Prepare Diffusion Feature
```
python prep_diff.py --data_root ./ --feat_path ./full_feat.npy --save_root ./
```
This will give you the following files:

* `full_diffusion.npy`: The diffused node features.
* `year_diffusion.npy`: The diffused year features.

## Step 7: MLP Training
```
# Fold k MLP Training (k from 0 to 4):
python train_mlp.py --fold k
```

## Step 8: MLP Inference
```
# Fold k MLP Inference (k from 0 to 4):
python infer_mlp.py --fold k
```

## Step 9: Ensemble All Fold Results
In our final submission, we take the average of five folds' results. Besides, to fully exploit the precious experiences of last year's winning team, we also train the R-Unimp model and ensemble it with our MLP's outputs.