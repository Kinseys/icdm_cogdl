# icdm_cogdl
cogdl version for icdm graph competition dataset

This repo provides a collection of cogdl baselines for DGraphFin dataset. Please download the dataset from the DGraph web and place & unzip it under the folder 'dataset/'  like: 'dataset/dgraphfin.npz'

ICDM introduction:https://tianchi.aliyun.com/competition/entrance/531976/introduction

**ICDM dataset:** https://tianchi.aliyun.com/competition/entrance/531976/information

**Cogdl introduction:** https://cogdl.readthedocs.io/en/latest/index.html

## Environments
Implementing environment:  
- numpy = 1.21.2  
- pytorch >= 1.6.0  
- pillow = 9.1.1
- cogdl = 0.5.3

## Training

- **MLP**
```bash
python gnn.py --model mlp --dataset ICDM --epochs 200 --runs 10 --device 0
```

- **GCN**
```bash
python gnn.py --model gcn --dataset ICDM --epochs 200 --runs 10 --device 0
```

- **GraphSAGE**
```bash
python gnn.py --model graphsage --dataset ICDM --epochs 200 --runs 10 --device 0
```

- **GIN**
```bash
python gnn.py --model gin --dataset ICDM --epochs 200 --runs 10 --device 0
```

- **GAT**
```bash
python gnn.py --model gat --dataset ICDM --epochs 200 --runs 10 --device 0
```

- **Grand**
```bash
python gnn.py --model grand --dataset ICDM --epochs 200 --runs 10 --device 0
```

- **SGC**
```bash
python gnn.py --model sgc --dataset ICDM --epochs 200 --runs 10 --device 0
```

- **SIGN**
```bash
python gnn.py --model sign --dataset ICDM --epochs 200 --runs 10 --device 0
```


- **You can find more models on cogdl https://cogdl.readthedocs.io/en/latest/index.html**


## Results:
Performance on **DGraphFin**(10 runs):

| Methods   | Train AUC  | Valid AUC  |
|  :----  |  ---- |  ---- | ---- |
| SIGN | 0.9421 ± 0.0031 | 0.9213 ± 0.0042 |
| GIN | 0.8965 ± 0.0257 | 0.8983 ± 0.0320 |
| GraphSAGE| 0.9213 ± 0.0022 | 0.8986 ± 0.0021 |
| GCN | 0.9374 ± 0.0011 | 0.8629 ± 0.0047 |
| MLP | 0.9035 ± 0.0033 | 0.7587 ± 0.0031 |
| Grand  | 0.6317 ± 0.0018 | 0.6292 ± 0.0051 |
| SGC | 0.6187 ± 0.0046 | 0.6136 ± 0.0043 |
