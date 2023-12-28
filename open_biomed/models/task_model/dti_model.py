import json
import random
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from transformers import AutoModel

from models import SUPPORTED_MOL_ENCODER, SUPPORTED_PROTEIN_ENCODER
from models.predictor import MLP
from feat.text_featurizer import name2tokenizer, name2model


class SparseAttention(nn.Module):
    def __init__(self, config, kge, mlp_dim):
        super(SparseAttention, self).__init__()
        self.config = config
        kge_id = list(kge.keys())
        kge_id.sort()
        kge_emb_list = [kge[k] for k in kge_id]
        self.kge_emb = torch.vstack(kge_emb_list)

        self.span_encoding = MLP({'hidden_size': [mlp_dim],
                                  'dropout': 0.0,
                                  'activation': 'relu',
                                  'batch_norm': False},
                                 mlp_dim,
                                 self.kge_emb[0].shape[0])

    def forward(self, x):
        spanned = self.span_encoding(x)
        self.kge_emb = self.kge_emb.to(spanned.dtype)
        dotprod = spanned @ self.kge_emb.T
        if len(dotprod.shape) == 1:
            dotprod = dotprod.unsqueeze(0)
        topmem, topmem_idx = torch.topk(dotprod, self.config['k'], dim=1)
        topmem = torch.exp(topmem)
        topmem = torch.nn.functional.softmax(topmem, dim=1)
        batch_size = topmem_idx.shape[0]
        topval = torch.stack([self.kge_emb[topmem_idx[i, :], :] for i in range(batch_size)])
        value = torch.vstack([topmem[i, :].unsqueeze(0) @ topval[i, :] for i in range(batch_size)])
        return value


class MultiHeadSparseAttention(nn.Module):
    def __init__(self, config, kge, mlp_dim):
        super(MultiHeadSparseAttention, self).__init__()
        n_heads = config['heads']
        kge_dim = kge[list(kge.keys())[0]].shape[0]
        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            self.heads.append(SparseAttention(config, kge, mlp_dim))
        self.linear = nn.Linear(n_heads * kge_dim, kge_dim)

    def forward(self, x):
        heads_output = [head(x) for head in self.heads]
        x = torch.hstack(heads_output)
        x = self.linear(x)
        return x


class DTIModel(nn.Module):
    def __init__(self, config, pred_dim):
        super(DTIModel, self).__init__()
        drug_encoder_config = json.load(open(config["drug"]["config_path"], "r"))
        self.drug_encoder = SUPPORTED_MOL_ENCODER[config["drug"]["name"]](drug_encoder_config)

        protein_encoder_config = json.load(open(config["protein"]["config_path"], "r"))
        self.protein_encoder = SUPPORTED_PROTEIN_ENCODER[config["protein"]["name"]](protein_encoder_config)

        self.pred_head = MLP(config["pred_head"], self.drug_encoder.output_dim + self.protein_encoder.output_dim,
                             pred_dim)

    def forward(self, drug, protein):
        h_drug = self.drug_encoder.encode_mol(drug)
        h_protein = self.protein_encoder.encode_protein(protein)
        h = torch.cat((h_drug, h_protein), dim=1)
        return self.pred_head(h)


class KEDD4DTI(nn.Module):
    def __init__(self, config, pred_dim):
        super(KEDD4DTI, self).__init__()
        self.config = config
        self.use_sparse_attention = config['sparse_attention']['active']
        self.projection_dim = config['projection_dim']
        self.kge = config['kge']
        self.kge_dim = self.kge[list(self.kge.keys())[0]].shape[0]

        if self.kge is None and self.use_sparse_attention:
            raise RuntimeError('No KGE to use for sparse attention')

        drug_encoder_config = json.load(open(config["drug"]["structure"]["config_path"], "r"))
        self.drug_structure_encoder = SUPPORTED_MOL_ENCODER[config["drug"]["structure"]["name"]](drug_encoder_config)
        if "init_checkpoint" in drug_encoder_config.keys():
            encoder_ckpt = drug_encoder_config["init_checkpoint"]
            assert encoder_ckpt
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            # self.encoder.load_state_dict(ckpt)
            missing_keys, unexpected_keys = self.drug_structure_encoder.load_state_dict(ckpt, strict=False)
            print("Encoder missing_keys: ", missing_keys)
            print("Encoder unexpected_keys: ", unexpected_keys)

        protein_encoder_config = json.load(open(config["protein"]["structure"]["config_path"], "r"))
        self.protein_structure_encoder = SUPPORTED_PROTEIN_ENCODER[config["protein"]["structure"]["name"]](
            protein_encoder_config)

        if self.use_sparse_attention:
            self.drug_mask_prob = config['sparse_attention']['drug_mask_prob']
            self.prot_mask_prob = config['sparse_attention']['prot_mask_prob']
            # Only used when KGE is not available (all zeros)
            kge_drug = {k: self.kge[k] for k in self.kge if k[0] == 'D'}
            kge_prot = {k: self.kge[k] for k in self.kge if k[0] != 'D'}
            self.sparse_attn_config = config['sparse_attention']
            self.drug_sparse_attn = MultiHeadSparseAttention(self.sparse_attn_config,
                                                             kge_drug,
                                                             self.drug_structure_encoder.output_dim)
            self.prot_sparse_attn = MultiHeadSparseAttention(self.sparse_attn_config,
                                                             kge_prot,
                                                             self.protein_structure_encoder.output_dim)

        self.structure_hidden_dim = self.drug_structure_encoder.output_dim + self.protein_structure_encoder.output_dim

        self.kg_project = nn.Sequential(
            nn.Linear(config["drug"]["kg"]["embedding_dim"] + config["protein"]["kg"]["embedding_dim"],
                      self.projection_dim),
            nn.Dropout(config["projection_dropout"])
        )

        self.text_project = nn.Sequential(
            nn.Linear(config["text_dim"], self.projection_dim),
            nn.Dropout(config["projection_dropout"])
        )

        self.pred_head = MLP(config["pred_head"], self.structure_hidden_dim + 2 * self.projection_dim, pred_dim)

        self.drug_kge_count = 0
        self.drug_nokge_count = 0
        self.prot_kge_count = 0
        self.prot_nokge_count = 0

    def forward(self, drug, protein):
        # drug is a dict {'structure' : torch_geometric.data.batch.DataBatch,
        #                 'kg' : torch.Tensor [B, 256],
        #                 'text' : _}
        # protein is a dict {'structure' : torch.Tensor [B, 1200],
        #                    'kg' : torch.Tensor [B, 256],
        #                    'text' : _}

        batch_size = drug['kg'].shape[0]
        for i in range(batch_size):
            if torch.any(drug['kg'][i, :]):
                self.drug_kge_count += 1
            else:
                self.drug_nokge_count += 1

            if torch.any(protein['kg'][i, :]):
                self.prot_kge_count += 1
            else:
                self.prot_nokge_count += 1

        h_drug_structure = self.drug_structure_encoder.encode_mol(drug["structure"])
        if self.config['drug']['structure']['name'] == 'graphmvp':
            h_drug_structure = h_drug_structure[0]
        h_protein_structure = self.protein_structure_encoder.encode_protein(protein["structure"])

        h_structure = torch.cat((h_drug_structure, h_protein_structure), dim=1)

        h_drug_kg = drug['kg']
        h_prot_kg = protein['kg']

        if self.use_sparse_attention:
            if self.training:
                for i in range(batch_size):
                    rand_drug = random.uniform(0, 1)
                    rand_prot = random.uniform(0, 1)
                    if rand_drug < self.drug_mask_prob:
                        h_drug_kg[i, :] = 0
                    if rand_prot < self.prot_mask_prob:
                        h_prot_kg[i, :] = 0

            drug_nokge = torch.all(h_drug_kg == 0, dim=1)
            drug_nokge = torch.nonzero(drug_nokge)
            if drug_nokge.numel() > 0:
                drug_nokge = torch.flatten(drug_nokge)
                h_drug_structure_subset = h_drug_structure[drug_nokge, :]
                if len(h_drug_structure_subset.shape) == 1:
                    h_drug_structure_subset = h_drug_structure_subset.unsqueeze(0)
                h_drug_nokge = self.drug_sparse_attn(h_drug_structure_subset)
                for i, j in enumerate(drug_nokge):
                    h_drug_kg[j, :] = h_drug_nokge[i, :]

            prot_nokge = torch.all(h_prot_kg == 0, dim=1)
            prot_nokge = torch.nonzero(prot_nokge)
            if prot_nokge.numel() > 0:
                prot_nokge = torch.flatten(prot_nokge)
                h_protein_structure_subset = h_protein_structure[prot_nokge, :]
                if len(h_protein_structure_subset.shape) == 1:
                    h_protein_structure_subset = h_protein_structure_subset.unsqueeze(0)
                h_prot_nokge = self.prot_sparse_attn(h_protein_structure_subset)
                for i, j in enumerate(prot_nokge):
                    h_prot_kg[j, :] = h_prot_nokge[i, :]

        h_kg = self.kg_project(torch.cat((h_drug_kg, h_prot_kg), dim=1))

        h_text = drug["text"]
        h_text = self.text_project(h_text)

        h = torch.cat((h_structure, h_kg, h_text), dim=1)
        return self.pred_head(h)

SUPPORTED_DTI_NETWORKS = {'deepeik': KEDD4DTI,
                          'mgraphdta': DTIModel,
                          'deepdta': DTIModel}
