import os
import json
from collections import defaultdict
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm
import torch
import dgl
import dgl.dataloading as dgldl
import dgl.sampling as dgl_sampling
import pytorch_lightning as pl

class GraphDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.g_full, self.id_to_idx, self.splits = None, None, None
        self.g_train = None
        self.rwr_train = None
        self.chem_gene_actions = []
        self.num_workers = hparams.num_workers

    def prepare_data(self):
        if os.path.exists(self.hparams.cache_graph) and not self.hparams.force_reload:
            print("--- Found all cached data artifacts. Skipping pre-processing. ---")
            with open(self.hparams.cache_actions, 'r') as f:
                self.chem_gene_actions = json.load(f)
            return

        print("--- [Rank 0] Running one-time data preparation (will be cached) ---")
        g_full, id_to_idx = self._load_and_build_graph()
        splits = self._split_edges(g_full)
        g_train = self._build_split_graphs(g_full, splits)
        rwr_train = self._compute_rwr_for_graph(g_train, "train")

        print("\n--- [Rank 0] Caching all data artifacts to disk ---")
        torch.save((g_full, id_to_idx), self.hparams.cache_graph)
        torch.save(splits, self.hparams.cache_splits)
        torch.save(g_train, self.hparams.cache_train_graph)
        torch.save(rwr_train, self.hparams.cache_rwr_train)
        with open(self.hparams.cache_actions, 'w') as f:
            json.dump(self.chem_gene_actions, f)
        print("✅ [Rank 0] Caching complete.")

    def setup(self, stage: Optional[str] = None):
        print(f"--- Loading cached graph data for setup ---")
        (self.g_full, self.id_to_idx) = torch.load(self.hparams.cache_graph)
        self.splits = torch.load(self.hparams.cache_splits)
        self.g_train = torch.load(self.hparams.cache_train_graph)
        self.rwr_train = torch.load(self.hparams.cache_rwr_train)
        with open(self.hparams.cache_actions, 'r') as f:
            self.chem_gene_actions = json.load(f)

        self.exclude_eids_val = {etype: eids for etype, eids in self.splits['val'].items() if eids.numel() > 0}
        self.exclude_eids_val.update({etype: eids for etype, eids in self.splits['test'].items() if eids.numel() > 0})
        self.exclude_eids_test = self.exclude_eids_val
        print(f"--- Data setup complete ---")

    def _shard_eids(self, eids_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'trainer'):
            return eids_dict
            
        world_size = self.trainer.world_size
        rank = self.trainer.global_rank
        sharded_eids = {}
        for etype, eids in eids_dict.items():
            if eids.numel() > 0:
                per_rank = eids.numel() // world_size
                start = rank * per_rank
                end = (rank + 1) * per_rank if rank != world_size - 1 else eids.numel()
                sharded_eids[etype] = eids[start:end]
        return sharded_eids

    def _build_loader(self, g, eids_dict, shuffle, num_workers, exclude_eids=None):
        eids_dict_filtered = {et: eids for et, eids in eids_dict.items() if eids.numel() > 0}
        sampler = dgldl.as_edge_prediction_sampler(
            dgldl.NeighborSampler([int(f) for f in self.hparams.fanouts.split(',')]),
            exclude=exclude_eids,
            negative_sampler=dgldl.negative_sampler.Uniform(self.hparams.neg_k)
        )
        return dgldl.DataLoader(
            g, eids_dict_filtered, sampler,
            batch_size=self.hparams.batch_size, shuffle=shuffle, drop_last=False,
            num_workers=num_workers,
            multiprocessing_context='spawn' if num_workers > 0 else None
        )

    def train_dataloader(self):
        train_eids_sharded = self._shard_eids(self.splits['train'])
        return self._build_loader(self.g_full, train_eids_sharded, True, self.num_workers, exclude_eids=self.exclude_eids_val)

    def val_dataloader(self):
        val_eids_sharded = self._shard_eids(self.splits['val'])
        return self._build_loader(self.g_full, val_eids_sharded, False, self.num_workers, exclude_eids=self.exclude_eids_val)

    def test_dataloader(self):
        test_eids_sharded = self._shard_eids(self.splits['test'])
        return self._build_loader(self.g_full, test_eids_sharded, False, self.num_workers, exclude_eids=self.exclude_eids_test)
        
    def predict_dataloader(self):
        test_eids_dict = {
            et: eids for et, eids in self.splits['test'].items()
            if et[0] == 'chemical' and et[2] == 'gene' and eids.numel() > 0
        }
        if not test_eids_dict:
            return []

        sampler = dgldl.as_edge_prediction_sampler(
            dgl.dataloading.NeighborSampler([-1]),
        )
        return dgldl.DataLoader(
            self.g_full, test_eids_dict, sampler,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            multiprocessing_context='spawn' if self.num_workers > 0 else None
        )

    def _load_and_build_graph(self):
        print("--- Starting Data Loading and Graph Construction ---")
        chunksize = self.hparams.chunksize
        file_paths = {
            'chem_gene_actions': os.path.join(self.hparams.base_data_path, 'CTD_chem_gene_ixns.csv'),
            'chemical_chemical': os.path.join(self.hparams.base_data_path, 'chemical_chemical_noNaN.csv'),
            'chemical_disease': os.path.join(self.hparams.base_data_path, 'CTD_chemicals_diseases.csv'),
            'chemical_pathway': os.path.join(self.hparams.base_data_path, 'CTD_chem_pathways_enriched.csv'),
            'gene_disease': os.path.join(self.hparams.base_data_path, 'CTD_genes_diseases.csv'),
            'gene_pathway': os.path.join(self.hparams.base_data_path, 'CTD_genes_pathways.csv'),
            'gene_gene': os.path.join(self.hparams.base_data_path, 'gene_gene.csv'),
        }
        id_sets = defaultdict(set)
        _clean_id = lambda val: str(int(float(val))) if pd.notna(val) else None
        print("Scanning all files for node discovery...")
        node_discovery_info = {
            'chem_gene_actions': ('chemical', 'gene', 'ChemicalID', 'GeneID'),
            'chemical_chemical': ('chemical', 'chemical', 'Chemical1_name1', 'Chemical2_name2'),
            'chemical_disease': ('chemical', 'disease', 'ChemicalID', 'DiseaseID'),
            'chemical_pathway': ('chemical', 'pathway', 'ChemicalID', 'PathwayID'),
            'gene_disease': ('gene', 'disease', 'GeneID', 'DiseaseID'),
            'gene_pathway': ('gene', 'pathway', 'GeneID', 'PathwayID'),
            'gene_gene': ('gene', 'gene', 'Gene 1', 'Gene 2'),
        }
        for key, (src_type, dst_type, src_col, dst_col) in tqdm(node_discovery_info.items(), desc="Scanning Files for Nodes"):
            file_path = file_paths.get(key)
            if not file_path or not os.path.exists(file_path): continue
            reader = pd.read_csv(file_path, usecols=[c for c in [src_col, dst_col] if c], chunksize=chunksize, low_memory=False)
            for chunk in reader:
                if src_col in chunk: id_sets[src_type].update(chunk[src_col].dropna().astype(str))
                if dst_col in chunk: id_sets[dst_type].update(chunk[dst_col].dropna().astype(str))
        
        id_to_idx = {ntype: {id_val: i for i, id_val in enumerate(sorted(ids))} for ntype, ids in id_sets.items()}
        for ntype, mapping in id_to_idx.items(): print(f"Found {len(mapping)} unique nodes for type: '{ntype}'")
        edge_data = defaultdict(list)
        print("\nProcessing files to build edges...")
        chem_gene_file = file_paths.get('chem_gene_actions')
        if chem_gene_file and os.path.exists(chem_gene_file):
            reader = pd.read_csv(chem_gene_file, usecols=['ChemicalID', 'GeneID', 'InteractionActions'], chunksize=chunksize, low_memory=False)
            for chunk in tqdm(reader, desc="Building Chemical-Gene Edges"):
                df_subset = chunk.dropna()
                df_subset['action'] = df_subset['InteractionActions'].str.split('|').str[0].str.lower().str.replace('[^a-z0-9]', '', regex=True)
                
                src_indices = df_subset['ChemicalID'].astype(str).map(id_to_idx['chemical'])
                tgt_indices = df_subset['GeneID'].astype(str).map(id_to_idx['gene'])
                
                valid_pairs = pd.concat([src_indices, tgt_indices, df_subset['action']], axis=1).dropna()
                valid_pairs.columns = ['src', 'tgt', 'action']
                
                for action_name, group in valid_pairs.groupby('action'):
                    if not action_name: continue
                    self.chem_gene_actions.append(action_name)
                    canonical_etype = ('chemical', action_name, 'gene')
                    edge_data[canonical_etype].extend(zip(group['src'].astype(int), group['tgt'].astype(int)))
        
        self.chem_gene_actions = sorted(list(set(self.chem_gene_actions)))
        print(f"Created {len(self.chem_gene_actions)} distinct chemical-gene relation types.")
        edge_mappings = [
            (('chemical', 'interacts_with_chemical', 'chemical'), 'chemical_chemical', 'Chemical1_name1', 'Chemical2_name2'),
            (('chemical', 'associated_with_disease', 'disease'), 'chemical_disease', 'ChemicalID', 'DiseaseID'),
            (('chemical', 'involved_in_pathway', 'pathway'), 'chemical_pathway', 'ChemicalID', 'PathwayID'),
            (('gene', 'associated_with_disease', 'disease'), 'gene_disease', 'GeneID', 'DiseaseID'),
            (('gene', 'participates_in_pathway', 'pathway'), 'gene_pathway', 'GeneID', 'PathwayID'),
            (('gene', 'interacts_with_gene', 'gene'), 'gene_gene', 'Gene 1', 'Gene 2'),
        ]
        for etype, key, src_col, dst_col in edge_mappings:
            file_path = file_paths.get(key)
            if not file_path or not os.path.exists(file_path): continue
            reader = pd.read_csv(file_path, usecols=[src_col, dst_col], chunksize=chunksize, low_memory=False)
            src_type, _, dst_type = etype
            for chunk in tqdm(reader, desc=f"Building {key} edges"):
                df_subset = chunk.dropna()
                src_indices = df_subset[src_col].astype(str).map(id_to_idx[src_type])
                tgt_indices = df_subset[dst_col].astype(str).map(id_to_idx[dst_type])
                valid_pairs = pd.DataFrame({'src': src_indices, 'tgt': tgt_indices}).dropna()
                if not valid_pairs.empty:
                    edge_data[etype].extend(zip(valid_pairs['src'].astype(int), valid_pairs['tgt'].astype(int)))
        print("\nDeduplicating edges and building graph...")
        final_edge_dict = {}
        for etype, eds in edge_data.items():
            if not eds: continue
            unique_edges = torch.unique(torch.tensor(eds, dtype=torch.long), dim=0)
            final_edge_dict[etype] = (unique_edges[:, 0], unique_edges[:, 1])

        num_nodes_dict = {nt: len(m) for nt, m in id_to_idx.items()}
        g = dgl.heterograph(final_edge_dict, num_nodes_dict=num_nodes_dict)
        print("\nGraph construction complete")
        print(g)
        return g, id_to_idx

    def _split_edges(self, g, train_ratio=0.8, val_ratio=0.1):
        print("\n--- Splitting Data ---")
        train_eids, val_eids, test_eids = {}, {}, {}
        chem_gene_etypes = [('chemical', action, 'gene') for action in self.chem_gene_actions]
        for etype in g.canonical_etypes:
            eids = torch.arange(g.num_edges(etype), dtype=torch.long)
            if etype in chem_gene_etypes:
                perm = torch.randperm(g.num_edges(etype))
                n = perm.numel()
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                train_eids[etype] = eids[perm[:n_train]]
                val_eids[etype] = eids[perm[n_train : n_train + n_val]]
                test_eids[etype] = eids[perm[n_train + n_val:]]
            else:
                train_eids[etype] = eids
        
        print("--- FINAL DATA SPLIT SUMMARY ---")
        print(f"Total Edges in Full Graph: {g.num_edges():,}")
        print(f"Total Edges in Training Set: {sum(e.numel() for e in train_eids.values()):,}")
        print(f"Total Edges in Validation Set: {sum(val_eids.get(et, torch.empty(0)).numel() for et in g.canonical_etypes):,}")
        print(f"Total Edges in Test Set: {sum(test_eids.get(et, torch.empty(0)).numel() for et in g.canonical_etypes):,}")
        return {'train': train_eids, 'val': val_eids, 'test': test_eids}

    def _build_split_graphs(self, g_full, splits):
        print("\n--- Building Training Message-Passing Graph ---")
        g_train = dgl.edge_subgraph(g_full, splits['train'], preserve_nodes=True)
        print("Training graph created.")
        print(g_train)
        return g_train

    def _compute_rwr_for_graph(self, g_mp, split_name):
        print(f"\n--- Computing RWR neighbors for '{split_name}' graph ---")
        hg = dgl.to_homogeneous(g_mp)
        ntypes_map = hg.ndata[dgl.NTYPE].cpu()
        nids_map = hg.ndata[dgl.NID].cpu()
        N, chunk = hg.num_nodes(), 4096
        homo_neighbors = {}
        for start in tqdm(range(0, N, chunk), desc="RWR neighbors (chunked)"):
            seeds = torch.arange(start, min(start + chunk, N), dtype=torch.int64)
            paths, _ = dgl_sampling.random_walk(hg, seeds, length=self.hparams.rwr_len, restart_prob=self.hparams.rwr_restart_prob)
            for i in range(len(seeds)):
                path = paths[i]
                valid_nodes = path[path != -1]
                if valid_nodes.numel() == 0: continue
                unique_nodes, counts = torch.unique(valid_nodes, return_counts=True)
                k = min(self.hparams.rwr_top_k, len(unique_nodes))
                top_k_indices = torch.topk(counts, k=k).indices
                homo_neighbors[seeds[i].item()] = unique_nodes[top_k_indices].tolist()
        out = {nt: defaultdict(list) for nt in g_mp.ntypes}
        for seed_hid, neigh_list in homo_neighbors.items():
            seed_type = g_mp.ntypes[ntypes_map[seed_hid].item()]
            seed_nid = nids_map[seed_hid].item()
            for nh_hid in neigh_list:
                ntype = g_mp.ntypes[ntypes_map[nh_hid].item()]
                nid = nids_map[nh_hid].item()
                out[seed_type][seed_nid].append((ntype, nid))
        return {ntype: dict(nid_map) for ntype, nid_map in out.items()}