"""Refactored dataset module for knowledge graph tasks."""

from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class MKG_Loader(Dataset):
    """Dataset class for knowledge graph completion tasks.

    Args:
        data (str): Name of the dataset (e.g. "DB15K")
        logger: Logger object for logging messages
        max_vis_len (int): Maximum number of visual features per entity

    Attributes:
        ent2id (Dict[str, int]): Entity to ID mapping
        id2ent (List[str]): ID to entity mapping
        rel2id (Dict[str, int]): Relation to ID mapping
        id2rel (List[str]): ID to relation mapping
        num_ent (int): Number of entities
        num_rel (int): Number of relations
        train (List[Tuple[int, int, int]]): Training triples
        valid (List[Tuple[int, int, int]]): Validation triples
        test (List[Tuple[int, int, int]]): Test triples
        filter_dict (Dict[Tuple, List[int]]): Filter dictionary for evaluation
    """

    def __init__(self, data: str, logger, max_vis_len: int = -1):
        """Initialize dataset by loading entities, relations and triples."""
        self.data = data
        self.logger = logger
        self.data_dir = f"data/{data}/"

        # Initialize mappings
        self.ent2id: Dict[str, int] = {}
        self.id2ent: List[str] = []
        self.rel2id: Dict[str, int] = {}
        self.id2rel: List[str] = []

        # Load entities
        self._load_entities()
        self.num_ent = len(self.ent2id)

        # Load relations
        self._load_relations()
        self.num_rel = len(self.rel2id)

        # Load triples
        self.train: List[Tuple[int, int, int]] = []
        self.valid: List[Tuple[int, int, int]] = []
        self.test: List[Tuple[int, int, int]] = []
        self._load_triples("train.txt", self.train)
        self._load_triples("valid.txt", self.valid)
        self._load_triples("test.txt", self.test)

        # Build filter dictionary
        self.filter_dict: Dict[Tuple, List[int]] = {}
        self._build_filter_dict()

    def _load_entities(self) -> None:
        """Load entities from entities.txt file."""
        try:
            with open(self.data_dir + "entities.txt", encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    entity = line.strip()
                    self.ent2id[entity] = idx
                    self.id2ent.append(entity)
        except FileNotFoundError:
            raise FileNotFoundError(f"entities.txt not found in {self.data_dir}")

    def _load_relations(self) -> None:
        """Load relations from relations.txt file."""
        try:
            with open(self.data_dir + "relations.txt", encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    relation = line.strip()
                    self.rel2id[relation] = idx
                    self.id2rel.append(relation)
        except FileNotFoundError:
            raise FileNotFoundError(f"relations.txt not found in {self.data_dir}")

    def _load_triples(self, filename: str, triple_list: List[Tuple[int, int, int]]) -> None:
        """Load triples from given file into specified list."""
        try:
            with open(self.data_dir + filename, encoding='utf-8') as f:
                for line in f:
                    h, r, t = line.strip().split("\t")
                    triple_list.append((
                        self.ent2id[h],
                        self.rel2id[r],
                        self.ent2id[t]
                    ))
        except FileNotFoundError:
            raise FileNotFoundError(f"{filename} not found in {self.data_dir}")

    def _build_filter_dict(self) -> None:
        """Build filter dictionary for evaluation."""
        for data_split in [self.train, self.valid, self.test]:
            for h, r, t in data_split:
                # For head prediction (-1, r, t)
                if (-1, r, t) not in self.filter_dict:
                    self.filter_dict[(-1, r, t)] = []
                self.filter_dict[(-1, r, t)].append(h)

                # For tail prediction (h, r, -1)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h, r, -1)] = []
                self.filter_dict[(h, r, -1)].append(t)

    def __len__(self) -> int:
        """Return number of training triples."""
        return len(self.train)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item for training.

        Returns:
            Tuple containing:
            - masked_triplet_h: [t, r + num_rel, num_ent] for head prediction
            - label_h: head entity ID
            - masked_triplet_t: [h, r, num_ent] for tail prediction
            - label_t: tail entity ID
        """
        h, r, t = self.train[idx]

        # Prepare masked triplets for head and tail prediction
        masked_triplet_h = [t, r + self.num_rel, self.num_ent]
        label_h = h
        masked_triplet_t = [h, r, self.num_ent]
        label_t = t

        return (
            torch.tensor(masked_triplet_h),
            torch.tensor(label_h),
            torch.tensor(masked_triplet_t),
            torch.tensor(label_t)
        )
