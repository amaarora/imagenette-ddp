import os
import torch
import argparse

import torchvision
from config import Config
import timm
import torch.nn as nn
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp


def train_fn(model, train_data_loader, optimizer, epoch, device_ids):
    model.train()
    fin_loss = 0.0
    tk = tqdm(train_data_loader, desc="Epoch" + " [TRAIN] " + str(epoch + 1))

    for t, data in enumerate(tk):
        data[0] = data[0].to(device_ids[0])
        data[1] = data[1].to(device_ids[0])

        optimizer.zero_grad()
        out = model(data[0])
        loss = nn.CrossEntropyLoss()(out, data[1])
        loss.backward()
        optimizer.step()

        fin_loss += loss.item()
        tk.set_postfix(
            {
                "loss": "%.6f" % float(fin_loss / (t + 1)),
                "LR": optimizer.param_groups[0]["lr"],
            }
        )
    return fin_loss / len(train_data_loader), optimizer.param_groups[0]["lr"]


def eval_fn(model, eval_data_loader, epoch, device_ids):
    model.eval()
    fin_loss = 0.0
    tk = tqdm(eval_data_loader, desc="Epoch" + " [VALID] " + str(epoch + 1))

    with torch.no_grad():
        for t, data in enumerate(tk):
            data[0] = data[0].to(device_ids[0])
            data[1] = data[1].to(device_ids[0])
            out = model(data[0])
            loss = nn.CrossEntropyLoss()(out, data[1])
            fin_loss += loss.item()
            tk.set_postfix({"loss": "%.6f" % float(fin_loss / (t + 1))})
        return fin_loss / len(eval_data_loader)


def train(local_world_size, local_rank):
    # sanity check
    if torch.cuda.device_count() <= 1:
        raise ValueError(
            f"This script only works in multi-gpu environment, \
            PyTorch detected {torch.cuda.device_count()} number of GPUs."
        )

    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))
    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    )

    # train and eval datasets
    train_dataset = torchvision.datasets.ImageFolder(
        Config["TRAIN_DATA_DIR"], transform=Config["TRAIN_AUG"]
    )
    eval_dataset = torchvision.datasets.ImageFolder(
        Config["TEST_DATA_DIR"], transform=Config["TEST_AUG"]
    )

    # train and eval dataloaders
    sampler = None
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config["BS"],
        sampler=sampler,
        shuffle=(sampler is None),
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=Config["BS"],
    )

    # model
    model = timm.create_model(Config["MODEL"], pretrained=Config["PRETRAINED"])
    model = model.cuda(device_ids[0])
    model = DistributedDataParallel(model, device_ids)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config["LR"])

    for epoch in range(Config["EPOCHS"]):
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        avg_loss_train, lr = train_fn(
            model, train_dataloader, optimizer, epoch, device_ids
        )
        avg_loss_eval = eval_fn(model, eval_dataloader, epoch, device_ids)
        print(
            f"EPOCH = {epoch} | TRAIN_LOSS = {avg_loss_train} | EVAL_LOSS = {avg_loss_eval}"
        )


def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    train(local_world_size, local_rank)

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()

    spmd_main(args.local_world_size, args.local_rank)
