from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

from collections import OrderedDict
import copy
import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from feat.drug_featurizer import SUPPORTED_DRUG_FEATURIZER, DrugMultiModalFeaturizer
from feat.protein_featurizer import SUPPORTED_PROTEIN_FEATURIZER, ProteinMultiModalFeaturizer
from utils.data_utils import balance_classes
from utils.mol_utils import can_smiles
from utils.kg_utils import SUPPORTED_KG, embed
from utils.split import kfold_split, cold_drug_split, cold_protein_split, cold_cluster_split


class DTIDataset(Dataset, ABC):
    def __init__(self, path, config, split_strategy, in_memory=True):
        super(DTIDataset, self).__init__()
        self.path = path
        self.config = config
        self.split_strategy = split_strategy
        self.in_memory = in_memory
        self._load_data()
        self._train_test_split()
        self.kge = None

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    @abstractmethod
    def _train_test_split(self):
        raise NotImplementedError

    def _build(self, eval_pair_index, save_path):
        # build after train / test datasets are determined for filtering out edges
        if len(self.config["drug"]["modality"]) > 1:
            kg_config = self.config["drug"]["featurizer"]["kg"]
            self.kg = SUPPORTED_KG[kg_config["kg_name"]](kg_config["kg_path"])
            self.drug2kg, self.drug2text, self.protein2kg, self.protein2text = self.kg.link(self)
            self.concat_text_first = self.config["concat_text_first"]
            filter_out = []
            for i_drug, i_protein in eval_pair_index:
                smi = self.smiles[i_drug]
                protein = self.proteins[i_protein]
                if smi in self.drug2kg and protein in self.protein2kg:
                    filter_out.append((self.drug2kg[smi], self.protein2kg[protein]))
            # embed once for consistency
            kge = embed(self.kg, 'ProNE', filter_out=filter_out, dim=kg_config["embed_dim"], save=True,
                        save_path=save_path)
            self.kge = kge
            self.config["drug"]["featurizer"]["kg"]["kge"] = kge
            self.config["protein"]["featurizer"]["kg"]["kge"] = kge
        else:
            self.concat_text_first = False
        self._configure_featurizer()
        # featurize all data pairs in one pass for training efficiency
        if self.in_memory:
            self._featurize()

    def index_select(self, indexes, balanced=False, keep_negatives=1.0, seed=42):
        """
        Creates new dataset objects by selecting a subset of indices
        :param indexes: indices of samples to select
        :param balanced: whether to balance classes (only relevant if self.type='classification')
        :param keep_balanced: probability in [0,1] to keep each negative sample (only relevant if self.type='classification')
        :param seed: seed to use for random sampling if balance=True
        :return: dataset with selected indices
        """
        assert keep_negatives >= 0.0 and keep_negatives <= 1.0
        new_dataset = copy.copy(self)
        new_dataset.pair_index = [self.pair_index[i] for i in indexes]
        new_dataset.labels = [self.labels[i] for i in indexes]
        if self.dataset_type == 'classification':
            if keep_negatives < 1.0:
                random.seed(seed)
                new_pair_index = []
                new_labels = []
                for i in range(len(new_dataset.labels)):
                    if new_dataset.labels[i] == 1:
                        append = True
                    elif new_dataset.labels[i] == 0:
                        if random.random() <= keep_negatives:
                            append = True
                        else:
                            append = False
                    else:
                        raise ValueError('Invalid label for classification dataset:', new_dataset.labels[i])
                    if append:
                        new_pair_index.append(new_dataset.pair_index[i])
                        new_labels.append(new_dataset.labels[i])
                new_dataset.pair_index = new_pair_index
                new_dataset.labels = new_labels
            if balanced:
                new_dataset.pair_index, new_dataset.labels = balance_classes(new_dataset.pair_index, new_dataset.labels, seed=seed)
        return new_dataset

    def _configure_featurizer(self):
        if len(self.config["drug"]["modality"]) > 1:
            self.drug_featurizer = DrugMultiModalFeaturizer(self.config["drug"])
            self.protein_featurizer = ProteinMultiModalFeaturizer(self.config["protein"])
            self.drug_featurizer.set_drug2kgid_dict(self.drug2kg)
            self.protein_featurizer.set_protein2kgid_dict(self.protein2kg)
            if not self.concat_text_first:
                self.drug_featurizer.set_drug2text_dict(self.drug2text)
                self.protein_featurizer.set_protein2text_dict(self.protein2text)
        else:
            drug_feat_config = self.config["drug"]["featurizer"]["structure"]
            self.drug_featurizer = SUPPORTED_DRUG_FEATURIZER[drug_feat_config["name"]](drug_feat_config)
            protein_feat_config = self.config["protein"]["featurizer"]["structure"]
            self.protein_featurizer = SUPPORTED_PROTEIN_FEATURIZER[protein_feat_config["name"]](protein_feat_config)

    def _featurize(self):
        logger.info("Featurizing...")
        self.featurized_drugs = []
        self.featurized_proteins = []
        for i_drug, i_protein in tqdm(self.pair_index):
            drug, protein = self.smiles[i_drug], self.proteins[i_protein]
            if len(self.config["drug"]["modality"]) > 1 and self.concat_text_first:
                processed_drug = self.drug_featurizer(drug, skip=["text"])
                processed_protein = self.protein_featurizer(protein)
                processed_drug["text"] = self.drug_featurizer["text"](
                    self.drug2text[drug] + " [SEP] " + self.protein2text[protein])
            else:
                processed_drug = self.drug_featurizer(drug)
                processed_protein = self.protein_featurizer(protein)
            self.featurized_drugs.append(processed_drug)
            self.featurized_proteins.append(processed_protein)

    def __getitem__(self, index):
        if not self.in_memory:
            drug, protein, label = self.smiles[self.pair_index[index][0]], self.proteins[self.pair_index[index][1]], \
            self.labels[index]
            processed_drug = self.drug_featurizer(drug)
            processed_protein = self.protein_featurizer(protein)
            if self.concat_text_first:
                processed_drug["text"] = self.drug_featurizer["text"](
                    self.drug2text[drug] + " [SEP] " + self.protein2text[protein])
            return processed_drug, processed_protein, label
        else:
            return self.featurized_drugs[index], self.featurized_proteins[index], self.labels[index]

    def __len__(self):
        return len(self.pair_index)


class DTIClassificationDataset(DTIDataset):
    def __init__(self, path, config, split_strategy):
        self.dataset_type = 'classification'
        super(DTIClassificationDataset, self).__init__(path, config, split_strategy)

    def _train_test_split(self):
        if self.split_strategy in ["warm", "cold_drug", "cold_protein"]:
            self.nfolds = 5
            if self.split_strategy == "warm":
                folds = kfold_split(len(self), 5)
            elif self.split_strategy == "cold_drug":
                folds = cold_drug_split(self, 5)
            else:
                folds = cold_protein_split(self, 5)
            self.folds = []
            for i in range(5):
                self.folds.append({
                    "train": np.concatenate(folds[:i] + folds[i + 1:], axis=0).tolist(),
                    "test": folds[i]
                })
        elif self.split_strategy == "cold_cluster":
            self.nfolds = 9
            self.folds = cold_cluster_split(self, 3)


class Yamanishi08(DTIClassificationDataset):
    def __init__(self, path, config, split_strategy):
        super(Yamanishi08, self).__init__(path, config, split_strategy)

    def _load_data(self):
        data = json.load(open(os.path.join(self.path, "drug.json")))
        self.smiles = [data[item]["SMILES"] for item in data]
        drugsmi2index = dict(zip(self.smiles, range(len(self.smiles))))

        data = json.load(open(os.path.join(self.path, "protein.json")))
        self.proteins = [data[item]["sequence"] for item in data]
        proteinseq2index = dict(zip(self.proteins, range(len(self.proteins))))

        df = pd.read_csv(os.path.join(self.path, "data.csv"))
        self.pair_index, self.labels = [], []

        for row in df.iterrows():
            row = row[1]
            self.pair_index.append(
                (drugsmi2index[row["compound_iso_smiles"]], proteinseq2index[row["target_sequence"]]))
            self.labels.append(int(row["affinity"]))
        logger.info("Yamanishi08's dataset, total %d samples" % (len(self)))


class BMKG_DTI(DTIClassificationDataset):
    def __init__(self, path, config, split_strategy):
        super(BMKG_DTI, self).__init__(path, config, split_strategy)

    def _load_data(self):
        data = json.load(open(os.path.join(self.path, "drug.json")))
        self.drug_keys = [item for item in data]
        self.smiles = [data[item]["SMILES"] for item in data]
        self.drugid2index = dict(zip(data.keys(), range(len(self.smiles))))

        data = json.load(open(os.path.join(self.path, "protein.json")))
        self.prot_keys = [item for item in data]
        self.proteins = [data[item]["sequence"] for item in data]
        self.proteinid2index = dict(zip(data.keys(), range(len(self.proteins))))

        df = pd.read_csv(os.path.join(self.path, "data.csv"))
        self.pair_index, self.labels = [], []

        for row in df.iterrows():
            row = row[1]
            self.pair_index.append((self.drugid2index[row["drug_id"]], self.proteinid2index[str(row["protein_id"])]))
            self.labels.append(int(row["affinity"]))

class MSSL2drug(DTIClassificationDataset):
    def __init__(self, path, config, split_strategy):
        super(MSSL2drug, self).__init__(path, config, split_strategy)

    def _load_data(self):
        with open(os.path.join(self.path, 'drug.txt'), 'r') as f:
            data = [line for line in f.read().split('\n') if line != '']
        self.drug_keys = [line[:7] for line in data]
        drug_keys2name = {line[:7] : line[8:] for line in data}
        with open(os.path.join(self.path, 'drug_smiles.json'), 'r') as f:
            drug_name2smiles = json.load(f)
        self.smiles = [drug_name2smiles[drug_keys2name[key]] for key in self.drug_keys]
        self.drugid2index = dict(zip(self.drug_keys, range(len(self.smiles))))

        with open(os.path.join(self.path, 'protein.txt'), 'r') as f:
            data = [line for line in f.read().split('\n') if line != '']
        self.prot_keys = []
        prot_keys2name = {}
        for line in data:
            colon_idx = line.find(':')
            self.prot_keys.append(line[:colon_idx])
            prot_keys2name[line[:colon_idx]] = line[colon_idx+1:]
        with open(os.path.join(self.path, 'protein_seqs.json'), 'r') as f:
            prot_name2seq = json.load(f)
        self.proteins = [prot_name2seq[prot_keys2name[key]] for key in self.prot_keys]
        self.proteinid2index = dict(zip(self.prot_keys, range(len(self.proteins))))

        with open(os.path.join(self.path, 'DTInet.txt'), 'r') as f:
            data = [line.split(' ') for line in f.read().split('\n') if line != '']

        self.pair_index, self.labels = [], []
        for i in range(len(data)):
            row = data[i]
            for j in range(len(row)):
                label = int(row[j])
                self.pair_index.append((i, j))
                self.labels.append(label)

class DTIRegressionDataset(DTIDataset):
    def __init__(self, path, config, split_strategy):
        self.dataset_type = 'regression'
        super(DTIRegressionDataset, self).__init__(path, config, split_strategy)


class Davis_KIBA(DTIRegressionDataset):
    def __init__(self, path, config, split_strategy):
        self.is_davis = "davis" in path
        self.is_kiba = 'kiba' in path
        super(Davis_KIBA, self).__init__(path, config, split_strategy)

    def _load_data(self):
        Y = pickle.load(open(os.path.join(self.path, "Y"), "rb"), encoding='latin1')
        label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)

        can_smis_dict = json.load(open(os.path.join(self.path, "ligands_can.txt")), object_pairs_hook=OrderedDict)
        can_smis = list(can_smis_dict.values())
        self.smiles = [can_smiles(smi) for smi in can_smis]

        proteins_dic = json.load(open(os.path.join(self.path, "proteins.txt")), object_pairs_hook=OrderedDict)
        self.proteins = list(proteins_dic.values())

        # data:
        self.pair_index = []
        self.labels = []
        train_folds = json.load(open(os.path.join(self.path, "folds/train_fold_setting1.txt")))
        for fold in train_folds:
            for i in fold:
                self.pair_index.append((label_row_inds[i], label_col_inds[i]))
                self.labels.append(Y[label_row_inds[i], label_col_inds[i]])
        self.train_index = list(range(len(self.labels)))
        test_fold = json.load(open(os.path.join(self.path, "folds/test_fold_setting1.txt")))
        for i in test_fold:
            self.pair_index.append((label_row_inds[i], label_col_inds[i]))
            self.labels.append(Y[label_row_inds[i], label_col_inds[i]])
        self.test_index = list(range(len(self.train_index), len(self.labels)))
        if self.is_davis:
            self.labels = [-float(np.log10(y / 1e9)) for y in self.labels]
        elif self.is_kiba:
            self.labels = [torch.tensor(label, dtype=torch.float32) for label in self.labels]
        else:
            raise RuntimeError

        logger.info("%s dataset, %d samples" % ("davis" if self.is_davis else "kiba", len(self)))

    def _train_test_split(self):
        self.val_index = []


SUPPORTED_DTI_DATASETS = {
    "yamanishi08": Yamanishi08,
    "bmkg-dti": BMKG_DTI,
    'mssl2drug': MSSL2drug,
    "davis": Davis_KIBA,
    "kiba": Davis_KIBA
}