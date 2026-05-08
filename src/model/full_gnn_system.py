from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import AUROC, AveragePrecision, BinaryAccuracy

from .full_gnn_model import FullGNNModel
from ..utils.metrics import average_precision_at_k, brier_score, expected_calibration_error, compute_ranking_metrics

class FullGNNSystem(pl.LightningModule):
    def __init__(self, g_meta, all_etypes, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = FullGNNModel(
            g_meta,
            self.hparams.feature_dim, self.hparams.hidden_dim, self.hparams.out_dim,
            self.hparams.num_heads, self.hparams.dropout, all_etypes
        )
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.train_auprc = AveragePrecision(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.test_auprc = AveragePrecision(task="binary")
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.full_features_cache = None

    def setup(self, stage: Optional[str] = None):
        if self.full_features_cache is None: self.update_full_features_cache()

    @torch.no_grad()
    def update_full_features_cache(self):
        print(f"\nUpdating full features cache on CPU...")
        self.model.eval()
        from copy import deepcopy
        temp_embeds = deepcopy(self.model.embeds).to('cpu')
        temp_proj = deepcopy(self.model.encoder.proj).to('cpu')
        self.full_features_cache = {
            nt: temp_proj[nt](temp_embeds[nt].weight).detach().cpu()
            for nt in self.model.embeds.keys()
        }
        self.model.train()
        print(f"CPU cache update complete.")

    def on_train_epoch_start(self):
        if self.current_epoch > 0: self.update_full_features_cache()

    def _common_step(self, batch):
        rwr_cache = self.trainer.datamodule.rwr_train
        _, pair_graph, neg_pair_graph, blocks = batch

        pos_logits, neg_logits = self.model(
            blocks, pair_graph, neg_pair_graph, rwr_cache, self.full_features_cache
        )

        losses, all_preds, all_labels = [], [], []
        ranking_pos_scores, ranking_neg_scores = [], []

        for rel, p_score in pos_logits.items():
            if p_score.numel() == 0: continue
            p_label = torch.ones_like(p_score)
            all_preds.append(p_score)
            all_labels.append(p_label)

            if neg_logits and rel in neg_logits and neg_logits[rel].numel() > 0:
                n_score = neg_logits[rel]
                n_label = torch.zeros_like(n_score)
                all_preds.append(n_score)
                all_labels.append(n_label)
                loss = F.binary_cross_entropy_with_logits(torch.cat([p_score, n_score]), torch.cat([p_label, n_label]))
            
                ranking_pos_scores.append(p_score)
                ranking_neg_scores.append(n_score)
            else:
                loss = F.binary_cross_entropy_with_logits(p_score, p_label)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True), None, None, None, None

        final_loss = torch.stack(losses).mean()
        all_preds_tensor = torch.cat(all_preds)
        all_labels_tensor = torch.cat(all_labels)
        return final_loss, all_preds_tensor, all_labels_tensor, ranking_pos_scores, ranking_neg_scores

    def training_step(self, batch, batch_idx):
        loss, preds, labels, _rp, _rn = self._common_step(batch)
        if preds is not None:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=labels.numel())
            self.training_step_outputs.append({'preds': preds.detach(), 'labels': labels.detach()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, _rp, _rn = self._common_step(batch)
        if preds is not None:
            self.log('val_loss', loss, prog_bar=True, sync_dist=True, batch_size=labels.numel())
            self.validation_step_outputs.append({'preds': preds.detach(), 'labels': labels.detach()})

    def test_step(self, batch, batch_idx):
        loss, preds, labels, rank_pos, rank_neg = self._common_step(batch)
        if preds is not None:
            entry = {'preds': preds.detach(), 'labels': labels.detach()}
         
            if rank_pos and rank_neg:
                entry['rank_pos'] = [t.detach() for t in rank_pos]
                entry['rank_neg'] = [t.detach() for t in rank_neg]
            self.test_step_outputs.append(entry)

    def on_train_epoch_end(self):
        if not self.training_step_outputs: return
        all_preds = torch.cat([x['preds'] for x in self.training_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.training_step_outputs])
        probs = torch.sigmoid(all_preds)
        labels_int = all_labels.int()

        self.log('train_auroc', self.train_auroc(probs, labels_int), on_epoch=True, sync_dist=True)
        self.log('train_auprc', self.train_auprc(probs, labels_int), on_epoch=True, sync_dist=True)
        self.log('train_acc', self.train_acc(probs, labels_int), on_epoch=True, sync_dist=True)
        self.log('train_ap10', average_precision_at_k(probs, all_labels, k=10), on_epoch=True, sync_dist=True)
        self.log('train_ap20', average_precision_at_k(probs, all_labels, k=20), on_epoch=True, sync_dist=True)
        self.log('train_brier', brier_score(probs, all_labels), on_epoch=True, sync_dist=True)
        self.log('train_ece', expected_calibration_error(probs, all_labels, n_bins=10), on_epoch=True, sync_dist=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        probs = torch.sigmoid(all_preds)
        labels_int = all_labels.int()

        self.log('val_auroc', self.val_auroc(probs, labels_int), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_auprc', self.val_auprc(probs, labels_int), on_epoch=True, sync_dist=True)
        self.log('val_acc', self.val_acc(probs, labels_int), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_ap10', average_precision_at_k(probs, all_labels, k=10), on_epoch=True, sync_dist=True)
        self.log('val_ap20', average_precision_at_k(probs, all_labels, k=20), on_epoch=True, sync_dist=True)
        self.log('val_brier', brier_score(probs, all_labels), on_epoch=True, sync_dist=True)
        self.log('val_ece', expected_calibration_error(probs, all_labels, n_bins=10), on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        if not self.test_step_outputs: return
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        probs = torch.sigmoid(all_preds)
        labels_int = all_labels.int()

        self.log('test_auroc', self.test_auroc(probs, labels_int), sync_dist=True)
        self.log('test_auprc', self.test_auprc(probs, labels_int), sync_dist=True)
        self.log('test_acc', self.test_acc(probs, labels_int), sync_dist=True)
        self.log('test_ap10', average_precision_at_k(probs, all_labels, k=10), sync_dist=True)
        self.log('test_ap20', average_precision_at_k(probs, all_labels, k=20), sync_dist=True)
        self.log('test_brier', brier_score(probs, all_labels), sync_dist=True)
        self.log('test_ece', expected_calibration_error(probs, all_labels, n_bins=10), sync_dist=True)

        
        rank_pos_all, rank_neg_all = [], []
        for entry in self.test_step_outputs:
            if 'rank_pos' in entry:
                rank_pos_all.extend(entry['rank_pos'])
                rank_neg_all.extend(entry['rank_neg'])
        neg_k = getattr(self.hparams, 'neg_k', 1)
        ranking = compute_ranking_metrics(rank_pos_all, rank_neg_all, neg_k)
        self.log('test_mrr', ranking['mrr'], sync_dist=True)
        self.log('test_hits@1', ranking['hits@1'], sync_dist=True)
        self.log('test_hits@3', ranking['hits@3'], sync_dist=True)
        self.log('test_hits@10', ranking['hits@10'], sync_dist=True)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)