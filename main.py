import argparse
import datetime
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

import wandb
from XD_Violence import XDViolence
from model.model import STPN
from utils import color, determinist_behavior, info

TCAM_THRESHOLD = 0.35


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Consistency-based Self-supervised Learning for Temporal Anomaly Localization")
    parser.add_argument("--name", type=str, default="CSL-TAL", help="Name of the experiment")

    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--output-path", type=str, default="./output")
    parser.add_argument("--pretrain", action="store_true", help="Pretrain the model without align loss")
    parser.add_argument("--save-model", action="store_true", help="Save the model at the end of training")
    parser.add_argument("--resume", type=str, default=None, help="Path to the checkpoint file")

    parser.add_argument("--num-samples", type=int, default=2, help="Number of samples to use")
    parser.add_argument("--max-seqlen", type=int, default=600, help="Maximum sequence length")

    parser.add_argument("--alpha", type=float, default=2e-8, help="Alpha parameter for smoothness loss")
    parser.add_argument("--beta", type=float, default=2e-3, help="Beta parameter for sparsity loss")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter for mse")
    parser.add_argument("--delay-align", type=int, default=0, help="Delay align loss (n_epochs)")
    parser.add_argument("--detach", action="store_true", help="Detach one of the samples in align loss")
    parser.add_argument("--pseudo-label", action="store_true", help="Use pseudo label")
    parser.add_argument("--median-percentile", type=float, default=0.5, help="Percentile for median")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--milestones", type=int, nargs="+", default=[30])
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--scheduler", type=str, default="step", choices=["cosine", "warmup", "step"])

    parser.add_argument("--transformer", type=int, default=0)
    parser.add_argument("--gated-attention", default=False, action="store_true")
    parser.add_argument("--sample-window", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--log", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    return parser.parse_args()


def interp(wt_cam: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    w = wt_cam.cpu().squeeze()
    zoom = 16 if w.ndim == 1 else (16, 1)
    f = scipy.ndimage.zoom(w, zoom, order=3)
    f[np.where(f < threshold)] = 0
    f[np.where(f > 1)] = 1
    return torch.tensor(f)


def get_proposal(interp_wt_cam: torch.Tensor, threshold: float = TCAM_THRESHOLD) -> List[tuple]:
    labels = torch.zeros_like(interp_wt_cam)
    labels[interp_wt_cam > threshold] = 1

    if labels.dim() == 1:
        labels = labels.unsqueeze(1)

    proposal = []
    for cls in range(labels.shape[1]):
        start_cls, stop_cls = [], []
        if labels[0, cls] == 1:
            start_cls.append(0)

        start_cls += (np.nonzero(np.ediff1d(labels[:, cls]) == 1)[0] + 1).tolist()
        stop_cls += np.nonzero(np.ediff1d(labels[:, cls]) == -1)[0].tolist()

        if labels[-1, cls] == 1:
            stop_cls.append(len(labels) - 1)

        proposal.append(list(zip(start_cls, stop_cls)))

    return proposal


def compute_scores(proposal: List[tuple], wt_cam) -> List[float]:
    if wt_cam.dim() == 1:
        wt_cam = wt_cam.unsqueeze(1)
    scores = torch.zeros_like(wt_cam)
    pred = []
    for cls in range(wt_cam.shape[1]):
        c_pred = []
        for start, stop in proposal[cls]:
            score = torch.sum((wt_cam[start:stop + 1, cls]) / (stop - start + 1))
            scores[start:stop + 1, cls] = score
            c_pred.append([cls, score.item(), start, stop])
        pred.append(c_pred)
    return scores, pred


def get_mask(features: torch.Tensor) -> torch.Tensor:
    mask = []
    total_len = sum([len(f) for f in features])
    lengths = [len(f) for f in features]
    if lengths == 1:
        return torch.ones(total_len, dtype=torch.bool)
    for i, f in enumerate(features):
        if len(mask) == 0:
            mask.append(torch.cat((torch.ones(len(f)), torch.zeros(total_len - len(f)))))
        elif len(mask) == len(features) - 1:
            mask.append(torch.cat((torch.zeros(len(f)), torch.ones(total_len - len(f)))))
        else:
            mask.append(torch.cat((torch.zeros(lengths[i - 1]),
                                   torch.ones(len(f)),
                                   torch.zeros(total_len - len(f) - lengths[i - 1]))))

    return torch.stack(mask).bool()


def train(model, data_loader, optimizer,
          device: torch.device, epoch: int, args) -> None:
    model.train()
    loss_fn = model.get_loss_fn(args)
    first_batch = True
    with tqdm(desc=f"[{epoch:2d}] Train", total=len(data_loader)) as pbar:

        for _, features, video_label, len_feat in data_loader:

            features = torch.cat(features).to(device)
            video_label = torch.stack(video_label).to(device)

            attention, classification = model(features, len_feat)
            loss, losses = loss_fn(classification, attention, video_label, features, epoch, len_feat, first_batch)
            first_batch = False

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{losses['class_loss']:.4f}",
                             l1=f"{losses['sparsity_loss']:.4f}", align=f"{losses['align_loss']:.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.3e}")

            if wandb.run:
                wandb.log({**{"train.loss": loss.item(), "lr": optimizer.param_groups[0]['lr'], "epoch": epoch},
                           **{f"train.{k}": v for k, v in losses.items()}})
    pbar.close()


def test(model, data_loader, device: torch.device, epoch: int,
         args, exp_name: str) -> List[float]:
    model.eval()
    loss_fn = model.get_loss_fn(args)

    scores = torch.tensor([])
    labels = torch.tensor([])

    attention_scores = []
    all_segment_labels = []
    video_name_labels = []
    video_name_frames = []
    all_interpolated_wt_cam = []
    all_interpolated_t_cam = []
    all_attentions = []
    frame_proposal_scores = torch.tensor([])
    frame_proposal_scores_tcam = torch.tensor([])
    all_frame_labels = torch.tensor([])
    final_result = {}
    final_result['results'] = {}

    with torch.no_grad(), tqdm(desc=f"[{epoch:2d}] Test", total=len(data_loader)) as pbar:
        for video_name, features, video_label, segment_labels, frame_labels in data_loader:

            features = features.to(device)
            video_label = video_label.to(device)

            attention, classification = model(features)
            loss, losses = loss_fn(classification, attention, video_label, features, epoch, len_feat=None)

            t_cam = model.t_cam(features)
            interpolated_t_cam = interp(t_cam)
            weighted_t_cam = t_cam * attention
            interpolated_wt_cam = interp(weighted_t_cam)
            proposal = get_proposal(interpolated_wt_cam)
            proposal_scores, ap_scores = compute_scores(proposal, interpolated_wt_cam)
            proposal_scores_tcam, _ = compute_scores(proposal, interpolated_t_cam)

            frame_proposal_scores = torch.cat((frame_proposal_scores, proposal_scores))
            frame_proposal_scores_tcam = torch.cat((frame_proposal_scores_tcam, proposal_scores_tcam))
            all_frame_labels = torch.cat((all_frame_labels, frame_labels), dim=1)
            all_interpolated_wt_cam.append(interpolated_wt_cam)
            all_interpolated_t_cam.append(interpolated_t_cam)
            scores = torch.cat((scores, classification.cpu()))
            labels = torch.cat((labels, video_label.cpu()))
            video_name_labels += [video_name] * features.shape[1]
            video_name_frames += [video_name] * frame_labels.shape[1]
            all_attentions.append(interp(attention, 0.0))
            attention_scores += weighted_t_cam.squeeze().cpu().numpy().tolist()
            all_segment_labels += segment_labels.squeeze().cpu().numpy().tolist()

            pbar.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}", class_loss=f"{losses['class_loss']:.4f}", sparsity=f"{losses['sparsity_loss']:.4f}")

            if wandb.run:
                wandb.log({**{"test.loss": loss.item(), "epoch": epoch},
                           **{f"test.{k}": v for k, v in losses.items()}})
    pbar.close()

    auc = roc_auc_score(labels, scores, average="micro")
    ap = average_precision_score(labels, scores, average="micro")
    ap_baseline = (sum(labels) / len(labels)).mean()

    all_attentions = torch.cat(all_attentions).tolist()
    all_frame_labels = all_frame_labels.squeeze()
    all_interpolated_wt_cam = torch.cat(all_interpolated_wt_cam).tolist()
    all_interpolated_t_cam = torch.cat(all_interpolated_t_cam).tolist()

    auc_raw_attention = roc_auc_score(all_frame_labels, all_attentions, average="micro")
    ap_raw_attention = average_precision_score(all_frame_labels, all_attentions, average="micro")

    auc_attention = roc_auc_score(all_segment_labels, attention_scores, average="micro")
    ap_attention = average_precision_score(all_segment_labels, attention_scores, average="micro")
    ap_baseline_segment = torch.tensor(all_segment_labels).view(-1).sum() / torch.tensor(all_segment_labels).view(-1).shape[0]

    auc_proposal = roc_auc_score(all_frame_labels, frame_proposal_scores, average="micro")
    ap_proposal = average_precision_score(all_frame_labels, frame_proposal_scores, average="micro")

    auc_proposal_tcam = roc_auc_score(all_frame_labels, frame_proposal_scores_tcam, average="micro")
    ap_proposal_tcam = average_precision_score(all_frame_labels, frame_proposal_scores_tcam, average="micro")

    auc_frame = roc_auc_score(all_frame_labels, all_interpolated_wt_cam, average="micro")
    ap_frame = average_precision_score(all_frame_labels, all_interpolated_wt_cam, average="micro")

    auc_frame_tcam = roc_auc_score(all_frame_labels, all_interpolated_t_cam, average="micro")
    ap_frame_tcam = average_precision_score(all_frame_labels, all_interpolated_t_cam, average="micro")
    ap_baseline_frame = all_frame_labels.sum() / len(all_frame_labels)

    if args.save_plots:
        df = pd.DataFrame({"video_name": video_name_frames,
                           "proposal_scores": frame_proposal_scores.squeeze().tolist(),
                           "frame_label": all_frame_labels.tolist(),
                           "interpolated_wt_cam": all_interpolated_wt_cam,
                           "interpolated_t_cam": all_interpolated_t_cam,
                           "all_attentions": all_attentions,
                           })

        df_proposal_group = df.groupby(['video_name'],
                                       as_index=False).agg({'frame_label': lambda x: list(x),
                                                            'proposal_scores': lambda x: list(x),
                                                            'interpolated_wt_cam': lambda x: list(x),
                                                            'interpolated_t_cam': lambda x: list(x),
                                                            'all_attentions': lambda x: list(x), }
                                                           )
        df_proposal_group["sequence_length"] = df_proposal_group.apply(lambda x: list(range(len(x.frame_label))), axis=1)
        df_proposal_group = df_proposal_group.sample(32)
        fig, axes = plt.subplots(nrows=len(df_proposal_group) // 4, ncols=4, figsize=(20, 40))
        for i, ax in enumerate(axes.flat):
            sns.lineplot(x="sequence_length", y="proposal_scores", data=df_proposal_group.iloc[i], ax=ax, color="blue")
            sns.lineplot(x="sequence_length", y="interpolated_wt_cam", data=df_proposal_group.iloc[i], ax=ax, color="green", alpha=0.4)
            sns.lineplot(x="sequence_length", y="interpolated_t_cam", data=df_proposal_group.iloc[i], ax=ax, color="orange", alpha=0.4)
            sns.lineplot(x="sequence_length", y="frame_label", data=df_proposal_group.iloc[i], ax=ax, color="red").axhline(0.1, color="black", linestyle="--")
            sns.lineplot(x="sequence_length", y="all_attentions", data=df_proposal_group.iloc[i], ax=ax, color="purple", alpha=0.5)
            ax.set_title(f'{df_proposal_group.iloc[i].video_name}, {epoch}')
            # ax.set_xlabel("Sequence length")
            # ax.set_ylabel("Attention score")
            ax.set_ylim(-0.1, 1.1)
            # ax.legend(["Attention score", "Attention labels"])
        plt.legend(["Proposal scores", "WT-CAM scores", "T-CAM scores", "Ground Truth", "Proposal threshold", "Raw Attention"])
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{exp_name}_proposal.png", bbox_inches="tight")

    if wandb.run:
        wandb.log({"test.auc": auc,
                   "test.ap": ap,
                   "test.auc_attention": auc_attention,
                   "test.ap_attention": ap_attention,
                   "test.auc_frame": auc_proposal,
                   "test.ap_frame": ap_proposal,
                   "test.auc_interpolated": auc_frame,
                   "test.ap_interpolated": ap_frame,
                   "test.plot": wandb.Image(plt),
                   "epoch": epoch, })
    plt.close()

    print(f"Test AUC: {color.CYAN}{auc:.4f}{color.END} - Test AP: {color.CYAN}{ap:.4f}{color.END} (baseline ap: {ap_baseline:.4f})")
    print(f"Test AUC raw_attention: {color.BLUE}{auc_raw_attention:.4f}{color.END} - ", end="")
    print(f"Test AP raw_attention: {color.BLUE}{ap_raw_attention:.4f}{color.END} (baseline ap: {ap_baseline_frame:.4f})")
    print(f"Test AUC attention: {color.RED}{auc_attention:.4f}{color.END} - ", end="")
    print(f"Test AP attention: {color.RED}{ap_attention:.4f}{color.END} (baseline ap attention: {ap_baseline_segment:.4f})")
    print(f"Test AUC frame proposal: {color.RED}{auc_proposal:.4f}{color.END} - ", end="")
    print(f"Test AP frame proposal: {color.RED}{ap_proposal:.4f}{color.END} (baseline ap frame: {ap_baseline_frame:.4f})")
    print(f"Test AUC frame tcam proposal: {color.BLUE}{auc_proposal_tcam:.4f}{color.END} - ", end="")
    print(f"Test AP frame tcam proposal: {color.BLUE}{ap_proposal_tcam:.4f}{color.END} (baseline ap frame: {ap_baseline_frame:.4f})")
    print(f"Test AUC frame interp: {color.RED}{auc_frame:.4f}{color.END} - ", end="")
    print(f"Test AP frame interp: {color.RED}{ap_frame:.4f}{color.END} (baseline ap frame: {ap_baseline_frame:.4f})")
    print(f"Test AUC frame tcam interp: {color.BLUE}{auc_frame_tcam:.4f}{color.END} - ", end="")
    print(f"Test AP frame tcam interp: {color.BLUE}{ap_frame_tcam:.4f}{color.END} (baseline ap frame: {ap_baseline_frame:.4f})")

    return auc, ap, auc_attention, ap_attention


def custom_collate(batch: torch.Tensor) -> List[torch.Tensor]:
    return list(zip(*batch))


def load_data(args):
    train_ds = XDViolence(data_path=args.data_path, test_mode=False, max_seqlen=args.max_seqlen,
                          sample_num=args.num_samples, sample_window=args.sample_window)
    test_ds = XDViolence(data_path=args.data_path, test_mode=True, sample_num=args.num_samples)
    return train_ds, test_ds


def main(args: argparse.Namespace) -> None:

    train_ds, test_ds = load_data(args)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                           shuffle=True, num_workers=4, collate_fn=custom_collate)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    model = STPN(in_features=train_ds.features_len, num_classes=train_ds.num_class,
                 transformer=args.transformer, gated_att=args.gated_attention)

    # Enable cuda and parallelism
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    info("Using single GPU") if device.type == "cuda" else info("Using CPU")
    model.to(device)

    # Select optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "warmup":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4,
                                                                         T_0=(args.epochs + 1) // 2, T_mult=1)

    # Set up wandb
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    align = "align_" if args.num_samples > 1 else "no_align_"
    exp_name = f"{args.name}_{align}{now}"
    info(f"Experiment name: {exp_name}")
    info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.log and not args.test_only:
        wandb.init(project="csl-tal", name=exp_name, config=args)
        wandb.watch(model)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer.param_groups[0]["lr"] = args.lr * 0.01
        scheduler.load_state_dict(checkpoint["scheduler"])
        info(f"Resumed from {args.resume}")

    if args.test_only:
        test(model, test_dl, device, -1, args, exp_name)
        return

    auc_max, ap_max, auc_attention_max, ap_attention_max = 0, 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        train(model, train_dl, optimizer, device, epoch, args)
        auc, ap, auc_attention, ap_attention = test(model, test_dl, device, epoch, args, exp_name)
        scheduler.step()

        auc_max = max(auc_max, auc)
        ap_max = max(ap_max, ap)
        auc_attention_max = max(auc_attention_max, auc_attention)
        ap_attention_max = max(ap_attention_max, ap_attention)

        if wandb.run:
            wandb.log({"max.auc": auc_max,
                       "max.ap": ap_max,
                       "max.auc_attention": auc_attention_max,
                       "max.ap_attention": ap_attention_max,
                       "epoch": epoch, })

    if args.save_model:
        checkpoint = {"model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
        os.makedirs(args.output_path, exist_ok=True)
        torch.save(checkpoint, f"{args.output_path}/{exp_name}.pth")


if __name__ == '__main__':
    determinist_behavior()
    args = parse_args()
    main(args)
