from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

import copy
import numpy as np
import json
import os.path as osp
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from feat.protein_featurizer import SUPPORTED_PROTEIN_FEATURIZER, ProteinMultiModalFeaturizer
from utils.kg_utils import embed, subgraph_sample, SUPPORTED_KG

class PPIDataset(Dataset, ABC):
    def __init__(self, path, config, directed=False, make_network=False, paired_text=False, split='random'):
        super(PPIDataset, self).__init__()
        self.path = path
        self.config = config
        self.directed = directed
        self.make_network = make_network
        self.paired_text = paired_text
        self.kge = None
        self.featurized = False
        self._load_proteins()
        self._load_ppis()
        self._train_test_split(strategy=split)

    @abstractmethod
    def _load_proteins(self):
        raise NotImplementedError

    @abstractmethod
    def _load_ppis(self):
        raise NotImplementedError

    def _build(self, eval_pair_index, save_path):
        # build after train / test datasets are determined for filtering out edges
        if len(self.config["modality"]) > 1:
            kg_config = self.config["featurizer"]["kg"]
            self.kg = SUPPORTED_KG[kg_config["kg_name"]](kg_config["kg_path"])
            _, _, self.prot2kg, self.prot2text = self.kg.link(self)

            filter_out = []
            for i_protA, i_protB in eval_pair_index:
                seqA = self.proteins[i_protA]
                seqB = self.proteins[i_protB]
                if seqA in self.prot2kg and seqB in self.prot2kg:
                    filter_out.append((self.prot2kg[seqA], self.prot2kg[seqB]))
            # embed once for consistency
            kge = embed(self.kg, 'ProNE', filter_out=filter_out, dim=kg_config["embed_dim"], save=True,
                        save_path=save_path)
            self.kge = kge
            self.config["featurizer"]["kg"]["kge"] = kge
        if not self.featurized:
            self._configure_featurizer()
            self._featurize()

    def _configure_featurizer(self):
        if len(self.config["modality"]) > 1:
            self.featurizer = ProteinMultiModalFeaturizer(self.config)
            self.featurizer.set_protein2kgid_dict(self.prot2kg)
        else:
            self.featurizer = SUPPORTED_PROTEIN_FEATURIZER[self.config["featurizer"]["structure"]["name"]](self.config["featurizer"]["structure"])

    def _featurize(self):
        logger.info("Featurizing...")
        if self.paired_text:
            self.featurized_protsA = []
            self.featurized_protsB = []
            for i_protA, i_protB in tqdm(self.pair_index):

                # self.featurized_protsA.append(torch.zeros(500,))
                # self.featurized_protsB.append(torch.zeros(500,))
                # continue

                protA, protB = self.proteins[i_protA], self.proteins[i_protB]
                if len(self.config["modality"]) > 1:
                    processed_protA = self.featurizer(protA, skip=['text'])
                    processed_protB = self.featurizer(protB)
                    processed_protA["text"] = self.featurizer["text"](
                        self.prot2text[protA] + " [SEP] " + self.prot2text[protB]
                    )
                else:
                    processed_protA = self.featurizer(protA)
                    processed_protB = self.featurizer(protB)
                self.featurized_protsA.append(processed_protA)
                self.featurized_protsB.append(processed_protB)
        else:
            self.featurized_prots = [self.featurizer(protein) for protein in tqdm(self.proteins)]
        self.labels = torch.tensor(self.labels, dtype=torch.float)
        self.featurized = True

    @abstractmethod
    def _train_test_split(self, strategy='random'):
        raise NotImplementedError

    def index_select(self, indexes, split='train'):
        new_dataset = copy.deepcopy(self)
        new_dataset.pair_index = [self.pair_index[i] for i in indexes]
        new_dataset.labels = [self.labels[i] for i in indexes]
        if self.make_network:
            if not self.featurized:
                self._configure_featurizer()
                self._featurize()
            if split == 'train':
                # inductive setting, remove edges in the test set during training
                new_dataset.network = Data(
                    x=torch.stack(self.featurized_prots),
                    edge_index=torch.tensor(np.array(new_dataset.pair_index).T, dtype=torch.long)
                )
            else:
                new_dataset.network = Data(
                    x=torch.stack(self.featurized_prots),
                    edge_index=torch.tensor(np.array(self.pair_index).T, dtype=torch.long)
                )
        return new_dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.paired_text:
            protA_id = self.pair_index[idx][0]
            protB_id = self.pair_index[idx][1]
            protA = self.proteins[protA_id]
            protB = self.proteins[protB_id]
            # return protA_id, protB_id, protA, protB, self.labels[idx], self.proteins
            return self.featurized_protsA[idx], self.featurized_protsB[idx], self.labels[idx]

        if not self.make_network:
            return self.featurized_prots[self.pair_index[idx][0]], self.featurized_prots[self.pair_index[idx][1]], self.labels[idx]
        else:
            return self.pair_index[idx][0], self.pair_index[idx][1], self.labels[idx]

class STRINGDataset(PPIDataset):
    def __init__(self, path, config, directed=False, make_network=False, paired_text=False, split='bfs'):
        super(STRINGDataset, self).__init__(path, config, directed, make_network, paired_text, split)
        self.num_classes = 7

    def _load_proteins(self):
        self.proteins = []
        self.protname2id = {}
        with open(osp.join(self.path, "sequences.tsv")) as f:
            for i, line in enumerate(f.readlines()):
                line = line.rstrip("\n").split("\t")
                self.protname2id[line[0]] = i
                self.proteins.append(line[1])
        logger.info("Num proteins: %d" % (len(self.proteins)))

    def _load_ppis(self):
        ppi_dict = {}
        class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5, 'expression': 6}
        with open(osp.join(self.path, "interactions.txt")) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.rstrip("\t").split("\t")
                prot1, prot2 = self.protname2id[line[0]], self.protname2id[line[1]]
                if not self.directed and prot1 > prot2:
                    t = prot1
                    prot1 = prot2
                    prot2 = t
                if (prot1, prot2) not in ppi_dict:
                    ppi_dict[(prot1, prot2)] = [0] * 7
                ppi_dict[(prot1, prot2)][class_map[line[2]]] = 1
        self.pair_index = []
        self.labels = []
        for prot_pair in ppi_dict:
            self.pair_index.append(list(prot_pair))
            self.labels.append(ppi_dict[prot_pair])
            if not self.directed:
                self.pair_index.append([prot_pair[1], prot_pair[0]])
                self.labels.append(ppi_dict[prot_pair])
        logger.info("Num ppis: %d" % (len(self.labels)))

    def _train_test_split(self, strategy='bfs', test_ratio=0.2, random_new=False):
        if random_new or not osp.exists(osp.join(self.path, "split.json")):
            self.test_indexes = subgraph_sample(len(self.proteins), self.pair_index, strategy, int(len(self.pair_index) * test_ratio), directed=False)
            self.train_indexes = []
            for i in range(len(self.pair_index)):
                if i not in self.test_indexes:
                    self.train_indexes.append(i)
            json.dump({
                "train": self.train_indexes,
                "test": self.test_indexes
            }, open(osp.join(self.path, "split_%s.json" % (strategy)), "w"))
        else:
            split = json.load(open(osp.join(self.path, "split_%s.json" % (strategy)), "r"))
            self.train_indexes = split["train"]
            self.test_indexes = split["test"]

SUPPORTED_PPI_DATASETS = {
    "SHS27k": STRINGDataset,
    "SHS148k": STRINGDataset,
    "STRING": STRINGDataset,
}