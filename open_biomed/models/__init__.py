from models.mol_encoder import *
from models.protein_encoder import *
from models.cell_encoder import *
from models.knowledge_encoder import *
from models.text_encoder import *
from models.multimodal_encoder import *

SUPPORTED_MOL_ENCODER = {
    "drug_cnn": DrugCNN,
    "tgsa": GINTGSA,
    "graphcl": GraphCL,
    "graphmvp": GraphMVP,
    "molclr": MolCLR,
    "mgnn": MGNN,
    "bert": MolBERT,
    "biomedgpt": BioMedGPT,
    "kv-plm": KVPLM,
    "momu": MoMu,
    "molfm": MolFM
}

SUPPORTED_PROTEIN_ENCODER = {
    "prot_cnn": ProtCNN,
    "cnn_gru": CNNGRU,
    "mcnn": MCNN,
    "pipr": CNNPIPR,
    "prottrans": ProtTrans
}

SUPPORTED_CELL_ENCODER = {
    "scbert": PerformerLM
}

SUPPORTED_TEXT_ENCODER = {
    "base_transformer": BaseTransformers,
    "biomedgpt": BioMedGPT,
    "kv-plm": KVPLM,
    "molfm": MolFM,
    "momu": MoMu,
    "text2mol": Text2MolMLP
}

SUPPORTED_KNOWLEDGE_ENCODER = {
    "TransE": TransE,
    "gin": GIN
}