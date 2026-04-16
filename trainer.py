"""
CEB-Net Trainer — based on CCFRec's CCFTrainer, adapted for multi-task losses.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torch import optim
from tqdm import tqdm
from colorama import init
from collections import defaultdict
from logging import getLogger
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir, set_color, get_local_time, log
from metrics import metrics_to_function

init(autoreset=True)


class CEBNetTrainer(object):
    def __init__(self, args, model, train_data, valid_data=None, test_data=None, device=None):
        self.args = args
        self.model = model
        self.logger = getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.eval_step = min(args.eval_step, self.epochs)
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.all_metrics = args.metrics.split(",")
        self.valid_metric = args.valid_metric
        self.max_topk = 0
        self.all_metric_name = []
        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            self.max_topk = max(self.max_topk, int(top_k))
            if m_name.lower() not in self.all_metric_name:
                self.all_metric_name.append(m_name.lower())

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.max_steps = self._get_train_steps()
        self.warmup_steps = args.warmup_steps
        self.optimizer = self._build_optimizer()
        if self.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps)
        else:
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps)

        self.device = device
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.dataset, args.save_file_name)
        ensure_dir(self.ckpt_dir)
        self.best_score = 0
        self.best_ckpt = "best_model.pth"

    def _build_optimizer(self):
        params = self.model.parameters()
        lr = self.lr
        wd = self.weight_decay
        name = self.learner.lower()
        if name == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        elif name == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=wd)
        else:
            return optim.Adam(params, lr=lr, weight_decay=wd)

    def _get_train_steps(self):
        n = len(self.train_data) // self.gradient_accumulation_steps
        return max(n, 1) * self.epochs

    def _train_epoch(self, epoch_idx, verbose=True):
        self.model.train()
        total_num = 0
        total_loss = defaultdict(float)

        iter_data = tqdm(
            self.train_data, total=len(self.train_data), ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"), disable=not verbose)

        for batch_idx, data in enumerate(iter_data):
            item_inters = data["item_inters"].to(self.device)
            inter_lens = data["inter_lens"].to(self.device)
            labels = data["targets"].to(self.device)
            code_inters = data["code_inters"].to(self.device)
            mask_labels = data["mask_targets"].to(self.device)

            total_num += 1
            self.optimizer.zero_grad()

            loss_dict = self.model.calculate_loss(
                item_inters, inter_lens, labels, code_inters, mask_labels)

            loss = loss_dict['loss']
            if torch.isnan(loss):
                raise ValueError("Training loss is nan")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            self.lr_scheduler.step()

            iter_data.set_postfix(loss=loss.item())
            for k in loss_dict:
                total_loss[k] += loss_dict[k].item()

        for k in total_loss:
            total_loss[k] /= total_num
        return total_loss

    def evaluate(self, scores, labels):
        metrics = {m: 0 for m in self.all_metrics}
        _, topk_idx = torch.topk(scores, self.max_topk, dim=-1)
        topk_idx = topk_idx.detach().cpu()
        labels = labels.detach().cpu()

        one_hot = torch.zeros_like(scores).detach().cpu()
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        top_k_labels = torch.gather(one_hot, dim=1, index=topk_idx).numpy()
        pos_nums = one_hot.sum(dim=1).numpy()

        topk_metrics = {}
        for m_name in self.all_metric_name:
            topk_metrics[m_name] = metrics_to_function[m_name](top_k_labels, pos_nums).sum(axis=0)

        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            metrics[m] = topk_metrics[m_name.lower()][int(top_k) - 1]
        return metrics

    def _save_checkpoint(self, epoch, verbose=True):
        path = os.path.join(self.ckpt_dir, self.best_ckpt)
        state = {
            "args": self.args, "epoch": epoch,
            "best_score": self.best_score,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path, pickle_protocol=4)
        if verbose:
            self.log(f"[Epoch {epoch}] Saving current: {path}")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss):
        out = f"[Epoch {epoch_idx}] training [time: {e_time - s_time:.2f}s, "
        if isinstance(loss, dict) or isinstance(loss, defaultdict):
            out += ", ".join(f"{k}: {v:.4f}" for k, v in loss.items())
        else:
            out += f"train loss: {loss:.4f}"
        return out + "]"

    def resume(self, ckpt_path):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_score = checkpoint.get('best_score', 0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        self.log(f"Resumed from {ckpt_path}, epoch {start_epoch}, best_score {self.best_score:.4f}")
        return start_epoch

    def fit(self, verbose=True, start_epoch=0):
        cur_eval_step = 0
        best_result = None
        for epoch_idx in range(start_epoch, self.epochs):
            t0 = time()
            train_loss = self._train_epoch(epoch_idx, verbose=verbose)
            t1 = time()

            if verbose:
                self.log(self._generate_train_loss_output(epoch_idx, t0, t1, train_loss))

            if (epoch_idx + 1) % self.eval_step == 0:
                metrics = self._test_epoch(test_data=self.valid_data, verbose=verbose)
                if metrics[self.valid_metric] > self.best_score:
                    self.best_score = metrics[self.valid_metric]
                    best_result = metrics
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                else:
                    cur_eval_step += 1
                if verbose:
                    self.log(f"[Epoch {epoch_idx}] Val Result: {metrics}")
                if cur_eval_step >= self.early_stop:
                    break

        return self.best_score, best_result

    @torch.no_grad()
    def test(self, verbose=True):
        if self.test_data is not None:
            return self._test_epoch(load_best_model=True, verbose=verbose)
        return None

    @torch.no_grad()
    def _test_epoch(self, test_data=None, load_best_model=False, verbose=True):
        if test_data is None:
            test_data = self.test_data

        if load_best_model:
            ckpt = os.path.join(self.ckpt_dir, self.best_ckpt)
            checkpoint = torch.load(ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            if verbose:
                self.log(f"Loading model from {ckpt}")

        self.model.eval()

        iter_data = tqdm(
            test_data, total=len(test_data), ncols=100,
            desc=set_color("Evaluate   ", "pink"), disable=not verbose)

        total = 0
        metrics = {m: 0 for m in self.all_metrics}
        for data in iter_data:
            item_inters = data["item_inters"].to(self.device)
            inter_lens = data["inter_lens"].to(self.device)
            code_inters = data["code_inters"].to(self.device)
            labels = data["targets"].to(self.device)

            total += len(labels)
            scores = self.model.full_sort_predict(item_inters, inter_lens, code_inters)

            _metrics = self.evaluate(scores, labels)
            for m, v in _metrics.items():
                metrics[m] += v

        for m in metrics:
            metrics[m] = metrics[m] / total
        return metrics

    def log(self, message, level='info'):
        return log(message, self.logger, level=level)
