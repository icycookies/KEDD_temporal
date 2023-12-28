###  

⚠️This repository is still under development. Unexpected errors could occur when running the code.

The codebase is built on [OpenBioMed](https://github.com/PharMolix/OpenBioMed). We are endavouring for a future release that incorporates the revised KEDD model.

To reproduce KEDD, install [BMKG](https://drive.google.com/drive/folders/1U2M3383-3dDAyLTAcXGcUagAEjlB6QgN?usp=sharing
) and put it under `assets/kg/`.

#### DTI

Yamanishi08 and BMKG_DTI can download from [here](https://drive.google.com/drive/folders/1AaUWLlOOua5BH7Q-bBVUBgOugDfWF3ip?usp=sharing). The 2 datasets should put under `datasets/dti/`. 

To run KEDD for drug-target interaction, use the following command:

```bash
bash scripts/dti/train_kedd.sh
```

#### DP

Download MoleculeNet datasets [here](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip), unzip the file, and put the dataset fold under `datasets/dp/`. You can use the following commands from within `OpenBioMed/`:

```shell
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mkdir -p datasets/dp
mv dataset datasets/dp/moleculenet
rm chem_dataset.zip
```

After downloading and unzipping, you should remove all the `processed/` directories of 8 datasets in the `dataset/` folder.

To run KEDD for drug property prediction, use the following command:

```bash
bash scripts/dp/train_kedd.sh
```

#### DDI

Luo's dataset is available [here](https://github.com/pengsl-lab/MSSL/blob/main/data/DownStreamdata/DDInet.txt). Put them under the `datasets/ddi/` folder.

To run KEDD for drug-drug interaction prediction, use the following command:

```bash
bash scripts/ddi/train_kedd.sh
```

#### PPI

The SHS27k and SHS148k datasets are available [here](https://github.com/lvguofeng/GNN_PPI/tree/main/data). Put them under `datasets/ppi/`

To run KEDD for protein-protein interaction prediction, use the following command:

```bash
bash scripts/ppi/run.sh
```

