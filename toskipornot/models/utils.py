import os

import numpy as np
import torch
from torchvision.utils import make_grid

import monai
from monai.inferers import sliding_window_inference
from monai.data import (
    list_data_collate,
    decollate_batch,
    DataLoader,
)

from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    SaveImage,
)
import wandb


def train_and_validate(
    model,
    device,
    optimizer,
    loss_function,
    config,
    train_files,
    val_files,
    train_transforms,
    val_transforms,
    output_folder,
    swin_size=256,
):

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        collate_fn=list_data_collate,
    )

    # start a typical PyTorch training
    max_epochs = config["max_epochs"]
    val_interval = config["val_interval"]
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if type(outputs) is list:
                outputs = outputs[0]
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            # 🐝 log train_loss for each step to wandb
            wandb.log({"train/loss": loss.item()})

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        # step scheduler after each epoch (cosine decay)
        # scheduler.step()

        # 🐝 log train_loss averaged over epoch to wandb
        wandb.log({"train/loss_epoch": epoch_loss})

        # 🐝 log learning rate after each epoch to wandb
        # wandb.log({"learning_rate": scheduler.get_lr()[0]})

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            best_metric, best_metric_epoch = validate_model(
                model,
                device,
                best_metric,
                best_metric_epoch,
                epoch,
                val_loader,
                output_folder,
                swin_size,
            )

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )

    # 🐝 log best score and epoch number to wandb
    wandb.log(
        {
            "best_dice_metric": best_metric,
            "best_metric_epoch": best_metric_epoch,
        }
    )
    return model


def validate_model(
    model,
    device,
    best_metric,
    best_metric_epoch,
    epoch,
    val_loader,
    output_folder,
    swin_size=256,
):
    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    model.eval()
    with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(
                device
            )
            if swin_size == 256:
                val_outputs = model(val_images)
            else:
                roi_size = (swin_size, swin_size)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_images, roi_size, sw_batch_size, model
                )

            if type(val_outputs) is list:
                val_outputs = val_outputs[0]

            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            image_caption = "Example validation output / label at epoch:" + str(epoch)
            val_image = torch.hstack((val_outputs[0], val_labels[0]))
            grid = make_grid(val_image[0])
            images = wandb.Image(grid[0, :, :], caption=image_caption)
            wandb.log({"Val output": images})

            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()
        os.makedirs(output_folder, exist_ok=True)
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                os.path.join(
                    output_folder, "best_metric_model_segmentation2d_dict.pth"
                ),
            )
            print("saved new best metric model")
        print(
            "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_epoch
            )
        )
        # 🐝 log validation dice score for each validation round
        wandb.log({"val/dice_metric": metric})
    return best_metric, best_metric_epoch


def test_model(
    model, device, config, test_files, data_transforms, output_folder, swin_size=256
):

    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])
    model.eval()

    test_ds = monai.data.Dataset(data=test_files, transform=data_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        collate_fn=list_data_collate,
    )

    output_saver = SaveImage(
        output_dir=output_folder, output_ext=".png", output_postfix="out"
    )
    label_saver = SaveImage(
        output_dir=output_folder, output_ext=".png", output_postfix="seg"
    )

    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data["img"].to(device), test_data["seg"].to(
                device
            )
            if swin_size == 256:
                test_outputs = model(test_images)
            else:
                roi_size = (swin_size, swin_size)
                sw_batch_size = 4
                test_outputs = sliding_window_inference(
                    test_images, roi_size, sw_batch_size, model
                )

            if type(test_outputs) is list:
                test_outputs = test_outputs[0]

            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            test_labels = [post_label(i) for i in decollate_batch(test_labels)]

            # compute metric for current iteration
            dice_metric(y_pred=test_outputs, y=test_labels)
            for test_output in test_outputs:
                output_saver(np.multiply(test_output, 255))
            for test_label in test_labels:
                label_saver(np.multiply(test_label, 255))
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        print("Test evaluation metric:", metric)
        # reset the status
        dice_metric.reset()
        # 🐝 log validation dice score for each validation round
        wandb.log({"test/dice_metric": metric})
