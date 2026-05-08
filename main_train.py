import os
import sys
import argparse
import json
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import time

import multiprocessing as mp
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import dgl
import dgl.function as fn
import dgl.sampling as dgl_sampling
import dgl.dataloading as dgldl

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.classification import AUROC, AveragePrecision, BinaryAccuracy

from torch_scatter import scatter_mean, scatter_softmax, scatter_add

from src.data.graph_datamodule import GraphDataModule
from src.model.full_gnn_system import FullGNNSystem

def main():
    parser = argparse.ArgumentParser("Stage 1: Train GNN for Multi-Relation Prediction")

    parser.add_argument('--base_data_path', type=str, required=True)
    parser.add_argument('--force_reload', action='store_true')
    parser.add_argument('--model_output_file', type=str, default='final_model_multi_rel.ckpt')
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--fanouts', type=str, default="15,10")
    parser.add_argument('--neg_k', type=int, default=1)
    parser.add_argument('--chunksize', type=int, default=800000)
    parser.add_argument('--rwr_len', type=int, default=30)
    parser.add_argument('--rwr_restart_prob', type=float, default=0.1)
    parser.add_argument('--rwr_top_k', type=int, default=20)
    parser.add_argument('--cache_graph', type=str, default='cached_graph_multi_rel.pt')
    parser.add_argument('--cache_splits', type=str, default='cached_splits_multi_rel.pt')
    parser.add_argument('--cache_train_graph', type=str, default='cached_train_graph_multi_rel.pt')
    parser.add_argument('--cache_rwr_train', type=str, default='cached_rwr_train_multi_rel.pt')
    parser.add_argument('--cache_actions', type=str, default='chem_gene_actions_list.json')

    args = parser.parse_args()
    pl.seed_everything(args.random_seed, workers=True)

    dm = GraphDataModule(args)

    is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"
    if is_main_process: dm.prepare_data()

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    if WORLD_SIZE > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("gloo", rank=int(os.environ.get("LOCAL_RANK", "0")), world_size=WORLD_SIZE)
        torch.distributed.barrier()

    g_train = torch.load(args.cache_train_graph)
    all_etypes = torch.load(args.cache_graph)[0].canonical_etypes
    model = FullGNNSystem(g_train, all_etypes, args)

    checkpoint_callback = ModelCheckpoint(monitor='val_auroc', mode='max', filename='best-model-multi-rel', save_top_k=1, verbose=True)
    early_stop_callback = EarlyStopping(monitor='val_auroc', patience=5, mode='max', verbose=True)

    strategy = 'ddp' if WORLD_SIZE > 1 else None

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices="auto", num_nodes=args.num_nodes, max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback], strategy=strategy,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)
    if trainer.is_global_zero: trainer.save_checkpoint(args.model_output_file)

    print("\n--- Training Finished. Testing on the best model for link prediction metrics... ---")
    test_results = trainer.test(datamodule=dm, ckpt_path='best')

    if trainer.is_global_zero:
        print("\n" + "="*80)
        print("### [Rank 0] Starting Final Evaluation and Prediction Export ###")
        print("="*80)

        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path or not os.path.exists(best_model_path):
            print(f"Best model checkpoint not found, using last saved model: {args.model_output_file}")
            best_model_path = args.model_output_file

        print(f"Loading best model from: {best_model_path}")

        dm_predict = GraphDataModule(args)
        dm_predict.setup('test')

        all_etypes = dm_predict.g_full.canonical_etypes
        best_model = FullGNNSystem.load_from_checkpoint(
            best_model_path, g_meta=dm_predict.g_train, all_etypes=all_etypes, hparams=args
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        best_model.to(device)
        best_model.eval()
        best_model.update_full_features_cache()

        idx_to_id = {ntype: {i: id_val for id_val, i in id_map.items()} for ntype, id_map in dm_predict.id_to_idx.items()}

        predict_loader = dm_predict.predict_dataloader()
        prediction_rows = []
        chem_gene_etypes = [('chemical', action, 'gene') for action in dm_predict.chem_gene_actions]

        total_correct = 0
        total_count = 0

        with torch.no_grad():
            for input_nodes, pair_graph, blocks in tqdm(predict_loader, desc="Predicting on Test Set"):
                blocks = [b.to(device) for b in blocks]
                pair_graph = pair_graph.to(device)

                node_embeds = best_model.model.encoder(
                    blocks,
                    {nt: best_model.model.embeds[nt](blocks[-1].srcnodes[nt].data[dgl.NID])
                     for nt in blocks[-1].srctypes},
                    best_model.full_features_cache, dm_predict.rwr_train
                )

                for etype in pair_graph.canonical_etypes:
                    u, v = pair_graph.edges(etype=etype)
                    if u.numel() == 0: continue

                    true_relation_name = etype[1]
                    total_count += u.numel()

                    h_u = node_embeds['chemical'][u]
                    h_v = node_embeds['gene'][v]

                    scores = []
                    for cg_etype in chem_gene_etypes:
                        key = '_'.join(cg_etype)
                        score = best_model.model.decoder.predictors[key](torch.cat([h_u, h_v], dim=1)).squeeze(1)
                        scores.append(score)

                    all_scores = torch.stack(scores, dim=1)
                    pred_indices = torch.argmax(all_scores, dim=1)

                    true_relation_index = dm_predict.chem_gene_actions.index(true_relation_name)
                    total_correct += (pred_indices == true_relation_index).sum().item()

                    pred_probs = torch.sigmoid(torch.max(all_scores, dim=1).values).cpu()
                    u_orig_idx = pair_graph.nodes['chemical'].data[dgl.NID][u].cpu()
                    v_orig_idx = pair_graph.nodes['gene'].data[dgl.NID][v].cpu()

                    for i in range(u.numel()):
                        prediction_rows.append({
                            "ChemicalID": idx_to_id['chemical'][u_orig_idx[i].item()],
                            "GeneID": idx_to_id['gene'][v_orig_idx[i].item()],
                            "Real_Relation": true_relation_name,
                            "Predicted_Relation": dm_predict.chem_gene_actions[pred_indices[i].item()],
                            "Prediction_Probability": pred_probs[i].item()
                        })

        pred_df = pd.DataFrame(prediction_rows)
        csv_path = "test_predictions_multi_relation.csv"
        pred_df.to_csv(csv_path, index=False)
        print(f"\n✅ Predictions for test set saved to '{csv_path}'")

        final_accuracy = total_correct / total_count if total_count > 0 else 0.0

        print("\n" + "="*80)
        print("###                           FINAL METRICS SUMMARY                           ###")
        print("="*80)

        print("\n---FINAL TRAINING---")
        train_auroc = trainer.callback_metrics.get("train_auroc", torch.tensor(-1.0)).item()
        train_auprc = trainer.callback_metrics.get("train_auprc", torch.tensor(-1.0)).item()
        train_acc = trainer.callback_metrics.get("train_acc", torch.tensor(-1.0)).item()
        train_ap10 = trainer.callback_metrics.get("train_ap10", torch.tensor(-1.0)).item()
        train_ap20 = trainer.callback_metrics.get("train_ap20", torch.tensor(-1.0)).item()
        print(f"Final train_auroc: {train_auroc:.4f}")
        print(f"Final train_auprc: {train_auprc:.4f}")
        print(f"Final train_accuracy: {train_acc:.4f}")
        print(f"Final train_ap10: {train_ap10:.4f}")
        print(f"Final train_ap20: {train_ap20:.4f}")

        print("\n--- BEST VALIDATION---")
        best_val_auroc = trainer.callback_metrics.get("val_auroc", torch.tensor(-1.0)).item()
        best_val_auprc = trainer.callback_metrics.get("val_auprc", torch.tensor(-1.0)).item()
        best_val_acc = trainer.callback_metrics.get("val_acc", torch.tensor(-1.0)).item()
        best_val_ap10 = trainer.callback_metrics.get("val_ap10", torch.tensor(-1.0)).item()
        best_val_ap20 = trainer.callback_metrics.get("val_ap20", torch.tensor(-1.0)).item()
        print(f"Best val_auroc: {best_val_auroc:.4f}")
        print(f"Best val_auprc: {best_val_auprc:.4f}")
        print(f"Best val_accuracy: {best_val_acc:.4f}")
        print(f"Best val_ap10: {best_val_ap10:.4f}")
        print(f"Best val_ap20: {best_val_ap20:.4f}")

        print("\n---FINAL TEST---")
        if test_results and isinstance(test_results, list) and isinstance(test_results[0], dict):
            test_metrics = test_results[0]
            test_auroc = test_metrics.get("test_auroc", -1.0)
            test_auprc = test_metrics.get("test_auprc", -1.0)
            test_acc = test_metrics.get("test_acc", -1.0)
            test_ap10 = test_metrics.get("test_ap10", -1.0)
            test_ap20 = test_metrics.get("test_ap20", -1.0)
            test_brier = test_metrics.get("test_brier", -1.0)
            test_ece = test_metrics.get("test_ece", -1.0)
            test_mrr = test_metrics.get("test_mrr", -1.0)
            test_hits1 = test_metrics.get("test_hits@1", -1.0)
            test_hits3 = test_metrics.get("test_hits@3", -1.0)
            test_hits10 = test_metrics.get("test_hits@10", -1.0)
            print(f"Test AUROC:    {test_auroc:.4f}")
            print(f"Test AUPRC:    {test_auprc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test AP@10:    {test_ap10:.4f}")
            print(f"Test AP@20:    {test_ap20:.4f}")
            print(f"Test Brier:    {test_brier:.4f}")
            print(f"Test ECE:      {test_ece:.4f}")
            print(f"Test MRR:      {test_mrr:.4f}")
            print(f"Test Hits@1:   {test_hits1:.4f}")
            print(f"Test Hits@3:   {test_hits3:.4f}")
            print(f"Test Hits@10:  {test_hits10:.4f}")
        else:
            print("Test metrics not available.")

        print("\n--- [FINAL TEST] Relation Prediction Metric ---")
        print(f"Relation Prediction Accuracy: {final_accuracy:.4f}")
        print("--------------------------------------------------\n")


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        try: mp.set_start_method('spawn', force=True)
        except RuntimeError: pass
    main()

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    if WORLD_SIZE > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("gloo", rank=int(os.environ.get("LOCAL_RANK", "0")), world_size=WORLD_SIZE)
        torch.distributed.barrier()

    g_train = torch.load(args.cache_train_graph)
    all_etypes = torch.load(args.cache_graph)[0].canonical_etypes
    model = FullGNNSystem(g_train, all_etypes, args)

    checkpoint_callback = ModelCheckpoint(monitor='val_auroc', mode='max', filename='best-model-multi-rel', save_top_k=1, verbose=True)
    early_stop_callback = EarlyStopping(monitor='val_auroc', patience=5, mode='max', verbose=True)
    
    strategy = 'ddp' if WORLD_SIZE > 1 else None

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices="auto", num_nodes=args.num_nodes, max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback], strategy=strategy,
        log_every_n_steps=10,
    )
    
    trainer.fit(model, datamodule=dm)
    if trainer.is_global_zero:
        trainer.save_checkpoint(args.model_output_file)
    
    print("\n--- Training Finished. Testing on the best model... ---")
    trainer.test(datamodule=dm, ckpt_path='best')

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    main()