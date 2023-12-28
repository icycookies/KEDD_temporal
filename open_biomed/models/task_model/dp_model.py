import json
import random
import torch
import torch.nn as nn

from transformers import AutoModel
from models import SUPPORTED_MOL_ENCODER
from models.multimodal_encoder.molfm.molfm import MolFM

activation = {
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
}


class MLP(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        hidden_dims = [input_dim] + config["hidden_size"] + [output_dim]
        for i in range(len(hidden_dims) - 1):
            self.model.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != len(hidden_dims) - 2:
                self.model.append(nn.Dropout(config["dropout"]))
                if config["activation"] != "none":
                    self.model.append(activation[config["activation"]])
                if config["batch_norm"]:
                    self.model.append(nn.BatchNorm1d())
    
    def forward(self, h):
        return self.model(h)


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


# TODO: choose header for different encoder
HEAD4ENCODER = {
    "deepeik": MLP,
    "momu": nn.Linear,
    "molfm": nn.Linear,
    "molclr": nn.Linear,
    "graphmvp": nn.Linear
}


class DPModel(nn.Module):

    def __init__(self, config, out_dim):
        super(DPModel, self).__init__()
        # prepare model
        if config["model"] == "KEDD":
            self.encoder = SUPPORTED_MOL_ENCODER[config["model"]](config["network"])
        elif config["model"] == "graphcl" or config["model"] == "molfm":
            self.encoder = SUPPORTED_MOL_ENCODER[config["model"]](config["network"]["structure"])
        else:
            self.encoder = SUPPORTED_MOL_ENCODER[config["model"]](**config["network"]["structure"])
        encoder_ckpt = config["network"]["structure"]["init_checkpoint"]
        if encoder_ckpt != "":
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            param_key = config["network"]["structure"]["param_key"]
            if param_key != "":
                ckpt = ckpt[param_key]
                missing_keys, unexpected_keys = self.encoder.load_state_dict(ckpt, strict=False)
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)
            
        self.proj_head = HEAD4ENCODER[config["network"]["structure"]["name"]](self.encoder.output_dim, out_dim)
        
    def forward(self, drug):
        if hasattr(self.encoder, "encode_structure") and not isinstance(self.encoder, MolFM):
            h = self.encoder.encode_structure(drug)  # Momu encoder_struct
        elif not isinstance(self.encoder, MolFM):
            h, _ = self.encoder(drug)  # encoder_struct
        else:
            h = self.encoder.encode_structure_with_kg(drug["structure"], drug["kg"])
        return self.proj_head(h)
    

class KEDD4DP(nn.Module):
    def __init__(self, config, out_dim):
        super(KEDD4DP, self).__init__()
        self.config = config
        self.use_sparse_attention = config['sparse_attention']['active']
        self.projection_dim = config["projection_dim"]
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
            print("encoder missing_keys: ", missing_keys)
            print("encoder unexpected_keys: ", unexpected_keys)

        if self.use_sparse_attention:
            self.drug_mask_prob = config['sparse_attention']['drug_mask_prob']
            # Only used when KGE is not available (all zeros)
            kge_drug = {k: self.kge[k] for k in self.kge if k[0] == 'D'}
            self.sparse_attn_config = config['sparse_attention']
            self.drug_sparse_attn = MultiHeadSparseAttention(self.sparse_attn_config,
                                                             kge_drug,
                                                             self.drug_structure_encoder.output_dim)

        self.structure_hidden_dim = self.drug_structure_encoder.output_dim
        
        self.kg_project = nn.Sequential(
                                        nn.Linear(config["drug"]["kg"]["embedding_dim"], self.projection_dim),
                                        nn.Dropout(config["projection_dropout"])
                                        )

        self.text_project = nn.Sequential(
            nn.Linear(config["text_dim"], self.projection_dim),
            nn.Dropout(config["projection_dropout"])
        )

        # structure + kg + text
        self.pred_head = MLP(config["pred_head"], self.structure_hidden_dim + 2 * self.projection_dim, out_dim)

        self.drug_kge_count = 0
        self.drug_nokge_count = 0
        
    def forward(self, drug):

        batch_size = drug['kg'].shape[0]
        for i in range(batch_size):
            if torch.any(drug['kg'][i, :]):
                self.drug_kge_count += 1
            else:
                self.drug_nokge_count += 1

        h_drug_structure = self.drug_structure_encoder(drug["structure"])
        if self.config['drug']['structure']['name'] == 'graphmvp':
            h_drug_structure = h_drug_structure[0]  # extracting h_graph from (h_graph, h_node)
        h_kg = drug["kg"]

        if self.use_sparse_attention:
            if self.training:
                for i in range(batch_size):
                    rand_drug = random.uniform(0, 1)
                    rand_prot = random.uniform(0, 1)
                    if rand_drug < self.drug_mask_prob:
                        h_kg[i, :] = 0
                    if rand_prot < self.prot_mask_prob:
                        h_kg[i, :] = 0

            drug_nokge = torch.all(h_kg == 0, dim=1)
            drug_nokge = torch.nonzero(drug_nokge)
            if drug_nokge.numel() > 0:
                drug_nokge = torch.flatten(drug_nokge)
                h_drug_structure_subset = h_drug_structure[drug_nokge, :]
                if len(h_drug_structure_subset.shape) == 1:
                    h_drug_structure_subset = h_drug_structure_subset.unsqueeze(0)
                h_drug_nokge = self.drug_sparse_attn(h_drug_structure_subset)
                for i, j in enumerate(drug_nokge):
                    h_kg[j, :] = h_drug_nokge[i, :]
        
        h_text = drug["text"]
        h_text = self.text_project(h_text)
        h = torch.cat((self.h_drug_structure, h_kg, h_text), dim=1)
        return self.pred_head(h)
