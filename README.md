# Distributed Training in PyTorch on ImageNette

This repository contains working code to train on ImageNette using [DISTRIBUTED DATA PARALLEL (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) in PyTorch and [Hugging Face Accelerate](https://github.com/huggingface/accelerate).

🤗 Accelerate - [DOCS](https://github.com/huggingface/accelerate) | [GitHub](https://github.com/huggingface/accelerate)

For a deep-dive into the HF Accelerate package, refer to [Inside Hugging Face's Accelerate!](https://wandb.ai/wandb_fc/pytorch-image-models/reports/Inside-Hugging-Face-s-Accelerate---Vmlldzo2MzgzNzA). 

To be able to run the scripts, please run the following commands first from the root directory of this repository to download the data: 

```
mkdir data && cd data 
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xvf imagenette2-160.tgz
```

Now you should have a `data` directory in the repository whose folder structure looks like:

```
data/
└── imagenette2-160
    ├── train
    │   ├── n01440764
    │   ├── n02102040
    │   ├── n02979186
    │   ├── n03000684
    │   ├── n03028079
    │   ├── n03394916
    │   ├── n03417042
    │   ├── n03425413
    │   ├── n03445777
    │   └── n03888257
    └── val
        ├── n01440764
        ├── n02102040
        ├── n02979186
        ├── n03000684
        ├── n03028079
        ├── n03394916
        ├── n03417042
        ├── n03425413
        ├── n03445777
        └── n03888257
```

## Launch training using PyTorch DDP 
To launch training using PyTorch DDP, run the following command from the `src` folder of this repository: 

```
./ddp.sh <number-of-gpus>
```

## Launch training using Huggingface Accelerate 
To launch training using Huggingface Accelerate, run the following command from the `src` folder of this repository: 

```
accelerate launch train_accelerate.py
```
