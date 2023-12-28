import json
import random
import torch
import torch.nn as nn

from models import SUPPORTED_PROTEIN_ENCODER, SUPPORTED_KNOWLEDGE_ENCODER
from models.predictor import MLP

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

class PPISeqModel(nn.Module):
    def __init__(self, config, num_classes):
        super(PPISeqModel, self).__init__()
        protein_encoder_config = json.load(open(config["encoder"]["config_path"], "r"))
        self.protein_encoder = SUPPORTED_PROTEIN_ENCODER[config["encoder"]["name"]](protein_encoder_config)
        self.feature_fusion = config["feature_fusion"]
        if self.feature_fusion == 'concat':
            in_dim = self.protein_encoder.output_dim * 2
        else:
            in_dim = self.protein_encoder.output_dim
        self.pred_head = MLP(config["pred_head"], in_dim, num_classes)

    def forward(self, prot1, prot2):
        x1 = self.protein_encoder(prot1)
        x2 = self.protein_encoder(prot2)
        #print(x1, x2)
        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        return self.pred_head(x)

class PPIGraphModel(nn.Module):
    def __init__(self, config, num_classes):
        super(PPIGraphModel, self).__init__()
        self.graph_encoder = SUPPORTED_KNOWLEDGE_ENCODER[config["name"]](config)
        self.feature_fusion = config["feature_fusion"]
        if self.feature_fusion == 'concat':
            in_dim = self.graph_encoder.output_dim * 2
        else:
            in_dim = self.graph_encoder.output_dim
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, prot1, prot2, graph):
        x = self.graph_encoder(graph)
        x1, x2 = x[prot1], x[prot2]
        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        x = self.fc(x)

        return x

class KEDD4PPI(nn.Module):
    def __init__(self, config, pred_dim):
        super(KEDD4PPI, self).__init__()
        self.config = config
        self.use_sparse_attention = config['sparse_attention']['active']
        self.projection_dim = config['projection_dim']
        self.kge = config['kge']
        self.kge_dim = self.kge[list(self.kge.keys())[0]].shape[0]

        if self.kge is None and self.use_sparse_attention:
            raise RuntimeError('No KGE to use for sparse attention')

        prot_encoder_config = json.load(open(config["protein"]["structure"]["config_path"], "r"))
        self.prot_structure_encoder = SUPPORTED_PROTEIN_ENCODER[config["protein"]["structure"]["name"]](prot_encoder_config)
        if "init_checkpoint" in prot_encoder_config.keys():
            encoder_ckpt = prot_encoder_config["init_checkpoint"]
            assert encoder_ckpt
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            # self.encoder.load_state_dict(ckpt)
            missing_keys, unexpected_keys = self.prot_structure_encoder.load_state_dict(ckpt, strict=False)
            print("Encoder missing_keys: ", missing_keys)
            print("Encoder unexpected_keys: ", unexpected_keys)

        if self.use_sparse_attention:
            self.mask_prob = config['sparse_attention']['mask_prob']
            # Only used when KGE is not available (all zeros)
            kge_prot = {k: self.kge[k] for k in self.kge if k[0] != 'D'}
            self.sparse_attn_config = config['sparse_attention']
            self.prot_sparse_attn = MultiHeadSparseAttention(self.sparse_attn_config,
                                                             kge_prot,
                                                             self.prot_structure_encoder.output_dim)
        self.structure_hidden_dim = 2 * self.prot_structure_encoder.output_dim
        self.kg_project = nn.Sequential(
            nn.Linear(2 * config["protein"]["kg"]["embedding_dim"],
                      self.projection_dim),
            nn.Dropout(config["projection_dropout"])
        )
        self.text_project = nn.Sequential(
            nn.Linear(config["text_dim"], self.projection_dim),
            nn.Dropout(config["projection_dropout"])
        )
        self.pred_head = MLP(config["pred_head"], self.structure_hidden_dim + 2 * self.projection_dim, pred_dim)

    def forward(self, protA, protB):
        batch_size = protA['kg'].shape[0]

        h_protA_structure = self.prot_structure_encoder.encode_protein(protA["structure"])
        if self.config['protein']['structure']['name'] == 'graphmvp':
            h_protA_structure = h_protA_structure[0]  # extracting h_graph from (h_graph, h_node)
        h_protB_structure = self.prot_structure_encoder.encode_protein(protB["structure"])
        if self.config['protein']['structure']['name'] == 'graphmvp':
            h_protB_structure = h_protB_structure[0]  # extracting h_graph from (h_graph, h_node)

        h_structure = torch.cat((h_protA_structure, h_protB_structure), dim=1)

        h_protA_kg = protA['kg']
        h_protB_kg = protB['kg']

        if self.use_sparse_attention:
            if self.training:
                for i in range(batch_size):
                    rand_protA = random.uniform(0, 1)
                    rand_protB = random.uniform(0, 1)
                    if rand_protA < self.mask_prob:
                        h_protA_kg[i, :] = 0
                    if rand_protB < self.mask_prob:
                        h_protB_kg[i, :] = 0

            protA_nokge = torch.all(h_protA_kg == 0, dim=1)
            protA_nokge = torch.nonzero(protA_nokge)
            if protA_nokge.numel() > 0:
                protA_nokge = torch.flatten(protA_nokge)
                h_protA_structure_subset = h_protA_structure[protA_nokge, :]
                if len(h_protA_structure_subset.shape) == 1:
                    h_protA_structure_subset = h_protA_structure_subset.unsqueeze(0)
                h_protA_nokge = self.prot_sparse_attn(h_protA_structure_subset)
                for i, j in enumerate(protA_nokge):
                    h_protA_kg[j, :] = h_protA_nokge[i, :]

            protB_nokge = torch.all(h_protB_kg == 0, dim=1)
            protB_nokge = torch.nonzero(protB_nokge)
            if protB_nokge.numel() > 0:
                protB_nokge = torch.flatten(protB_nokge)
                h_protB_structure_subset = h_protB_structure[protB_nokge, :]
                if len(h_protB_structure_subset.shape) == 1:
                    h_protB_structure_subset = h_protB_structure_subset.unsqueeze(0)
                h_protB_nokge = self.prot_sparse_attn(h_protB_structure_subset)
                for i, j in enumerate(protB_nokge):
                    h_protB_kg[j, :] = h_protB_nokge[i, :]

        h_kg = self.kg_project(torch.cat((h_protA_kg, h_protB_kg), dim=1))

        h_text = protA["text"]
        h_text = self.text_project(h_text)

        h = torch.cat((h_structure, h_kg, h_text), dim=1)
        return self.pred_head(h)
