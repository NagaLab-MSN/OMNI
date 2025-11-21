import os
import argparse
import multiprocessing as mp

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
import pandas as pd

from src.data.graph_datamodule import GraphDataModule
from src.model.full_gnn_system import FullGNNSystem

def main():
    parser = argparse.ArgumentParser("Stage 1: Train GNN for Multi-Relation Prediction")
    
    parser.add_argument('--base_data_path', type=str, required=True)
    parser.add_argument('--force_reload', action='store_true')
    parser.add_argument('--model_output_file', type=str, default='final_model_multi_rel.ckpt')
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=64)
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
    parser.add_argument('--rwr_len', type=int, default=10)
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
    if is_main_process:
        dm.prepare_data()

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