import argparse
import os
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import BBox8PoseDataset, collate_bbox8
from .heatmap import decode_heatmaps_argmax
from .losses import WeightedHeatmapMSELoss
from .metrics import corner_l2_error
from .model import BBox8PoseNet
from .utils import draw_corners, ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-image bbox8 heatmap network.")
    parser.add_argument("--labels_root", required=True, help="Path to bbox8_labels_obj_xxxxxx")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--heatmap_width", type=int, default=64)
    parser.add_argument("--heatmap_height", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--backbone", choices=["simple", "resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--pretrained_backbone", action="store_true", help="Use torchvision pretrained weights if available.")
    parser.add_argument("--base_channels", type=int, default=32, help="Only used when backbone=simple.")
    parser.add_argument("--decoder", choices=["fpn", "boxdreamer_lite"], default="boxdreamer_lite")
    parser.add_argument("--decoder_dim", type=int, default=192)
    parser.add_argument("--decoder_depth", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--vis_every", type=int, default=1, help="Export validation visualizations every N epochs.")
    parser.add_argument("--vis_num_samples", type=int, default=4, help="Number of validation samples to visualize.")
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    image_size = (args.image_width, args.image_height)
    heatmap_size = (args.heatmap_height, args.heatmap_width)

    train_set = BBox8PoseDataset(
        labels_root=args.labels_root,
        split="train",
        image_size=image_size,
        heatmap_size=heatmap_size,
        sigma=args.sigma,
        augment=True,
    )
    val_set = BBox8PoseDataset(
        labels_root=args.labels_root,
        split="val",
        image_size=image_size,
        heatmap_size=heatmap_size,
        sigma=args.sigma,
        augment=False,
    )
    return {
        "train": DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_bbox8,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_bbox8,
            pin_memory=True,
        ),
    }


def run_epoch(model, loader, criterion, optimizer, device, image_size, train: bool):
    model.train(train)
    total_loss = 0.0
    total_metric = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        target_heatmaps = batch["heatmaps"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        gt_xy = batch["corners_xy"].to(device)

        with torch.set_grad_enabled(train):
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, target_heatmaps, valid_mask)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        pred_xy = decode_heatmaps_argmax(pred_heatmaps.detach(), image_size=image_size)
        metrics = corner_l2_error(pred_xy.cpu(), gt_xy.cpu(), valid_mask.cpu())
        total_loss += loss.item()
        total_metric += metrics["mean_corner_l2"]
        num_batches += 1

    denom = max(1, num_batches)
    return {
        "loss": total_loss / denom,
        "mean_corner_l2": total_metric / denom,
    }


def export_val_visualizations(model, dataset, device, image_size, output_dir, epoch: int, num_samples: int) -> None:
    vis_dir = os.path.join(output_dir, "val_vis", f"epoch_{epoch:03d}")
    ensure_dir(vis_dir)

    model.eval()
    max_count = min(len(dataset), max(1, int(num_samples)))
    indices = np.linspace(0, len(dataset) - 1, num=max_count, dtype=int)

    for vis_idx, data_idx in enumerate(indices.tolist()):
        sample = dataset[data_idx]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_heatmaps = model(image)
            pred_xy = decode_heatmaps_argmax(pred_heatmaps, image_size=image_size)[0].cpu().numpy()

        gt_xy = sample["corners_xy"].cpu().numpy()
        valid_mask = sample["valid_mask"].cpu().numpy()
        rgb_path = sample["rgb_path"]
        image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]
        pred_xy_orig = pred_xy.copy()
        pred_xy_orig[:, 0] *= orig_w / float(image_size[0])
        pred_xy_orig[:, 1] *= orig_h / float(image_size[1])
        gt_xy_orig = sample["corners_xy_orig"].cpu().numpy()

        pred_vis = draw_corners(image_rgb, pred_xy_orig, valid_mask)
        gt_vis = draw_corners(image_rgb, gt_xy_orig, valid_mask)
        merged = np.concatenate([gt_vis, pred_vis], axis=1)
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        out_path = os.path.join(vis_dir, f"{vis_idx:02d}_{stem}_gt_pred.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    loaders = build_dataloaders(args)
    image_size = (args.image_width, args.image_height)

    device = torch.device(args.device)
    model = BBox8PoseNet(
        backbone=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        base_channels=args.base_channels,
        decoder=args.decoder,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = WeightedHeatmapMSELoss()

    best_val = float("inf")
    best_metric = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, loaders["train"], criterion, optimizer, device, image_size, train=True)
        val_stats = run_epoch(model, loaders["val"], criterion, optimizer, device, image_size, train=False)
        scheduler.step()

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_mean_corner_l2": train_stats["mean_corner_l2"],
            "val_loss": val_stats["loss"],
            "val_mean_corner_l2": val_stats["mean_corner_l2"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_stats)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_stats['loss']:.6f} "
            f"val_loss={val_stats['loss']:.6f} "
            f"val_corner_l2={val_stats['mean_corner_l2']:.3f}"
        )

        last_ckpt = os.path.join(args.output_dir, "last.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, last_ckpt)

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_ckpt = os.path.join(args.output_dir, "best.pt")
            torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, best_ckpt)

        if val_stats["mean_corner_l2"] < best_metric:
            best_metric = val_stats["mean_corner_l2"]
            best_metric_ckpt = os.path.join(args.output_dir, "best_metric.pt")
            torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, best_metric_ckpt)

        if epoch % args.save_every == 0:
            epoch_ckpt = os.path.join(args.output_dir, f"epoch_{epoch:03d}.pt")
            torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, epoch_ckpt)

        if epoch % args.vis_every == 0:
            export_val_visualizations(
                model=model,
                dataset=loaders["val"].dataset,
                device=device,
                image_size=image_size,
                output_dir=args.output_dir,
                epoch=epoch,
                num_samples=args.vis_num_samples,
            )

    save_json(os.path.join(args.output_dir, "train_history.json"), {"history": history})


if __name__ == "__main__":
    main()
