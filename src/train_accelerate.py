import os
import torch
import argparse

import torchvision
from config import Config
import timm
import torch.nn as nn
from tqdm import tqdm
from accelerate import Accelerator


def train_fn(model, train_data_loader, optimizer, epoch, accelerator):
    model.train()
    fin_loss = 0.0
    tk = tqdm(
        train_data_loader,
        desc="Epoch" + " [TRAIN] " + str(epoch + 1),
        disable=not accelerator.is_local_main_process,
    )

    for t, data in enumerate(tk):
        optimizer.zero_grad()
        out = model(data[0])
        loss = nn.CrossEntropyLoss()(out, data[1])
        accelerator.backward(loss)
        optimizer.step()

        fin_loss += loss.item()
        tk.set_postfix(
            {
                "loss": "%.6f" % float(fin_loss / (t + 1)),
                "LR": optimizer.param_groups[0]["lr"],
            }
        )
    return fin_loss / len(train_data_loader), optimizer.param_groups[0]["lr"]


def eval_fn(model, eval_data_loader, epoch, accelerator):
    model.eval()
    fin_loss = 0.0
    tk = tqdm(
        eval_data_loader,
        desc="Epoch" + " [VALID] " + str(epoch + 1),
        disable=not accelerator.is_local_main_process,
    )

    with torch.no_grad():
        for t, data in enumerate(tk):
            out = model(data[0])
            loss = nn.CrossEntropyLoss()(out, data[1])
            fin_loss += loss.item()
            tk.set_postfix({"loss": "%.6f" % float(fin_loss / (t + 1))})
        return fin_loss / len(eval_data_loader)


def train():
    accelerator = Accelerator()

    # train and eval datasets
    train_dataset = torchvision.datasets.ImageFolder(
        Config["TRAIN_DATA_DIR"], transform=Config["TRAIN_AUG"]
    )
    eval_dataset = torchvision.datasets.ImageFolder(
        Config["TEST_DATA_DIR"], transform=Config["TEST_AUG"]
    )

    # train and eval dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config["BS"]
    )
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=Config["BS"])

    # model
    model = timm.create_model(Config["MODEL"], pretrained=Config["PRETRAINED"])

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config["LR"])

    # prepare for DDP
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    for epoch in range(Config["EPOCHS"]):
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        avg_loss_train, lr = train_fn(
            model, train_dataloader, optimizer, epoch, accelerator
        )
        avg_loss_eval = eval_fn(model, eval_dataloader, epoch, accelerator)
        accelerator.print(
            f"EPOCH = {epoch} | TRAIN_LOSS = {avg_loss_train} | EVAL_LOSS = {avg_loss_eval}"
        )


if __name__ == "__main__":
    train()
