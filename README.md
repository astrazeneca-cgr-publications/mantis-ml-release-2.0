# Mantis-ML 2.0


- [Introduction](#introduction) 
- [Installation](#installation) 
- [Run](#run) 



Introduction
============
We previously introduced Mantis-ML, an automated machine-learning framework able to generalize from a set of known associated genes, identifying disease-gene association levels exome-wide based on publicly available gene annotation data.

We leverage knowledge graphs, coupled with graph neural networks to exploit the latent structure of associations between genes to further identify diesase-gene associations.

Additionally, we employ Natural Language Processing (NLP) techniques, specifically BioWordVec word embeddings, that identify and incorporate disease-relevant features based on semantic similarity. 

Collectively, this has enabled the development of a fully streamlined, automated, and scalable Mantis-ML workflow phenome-wide. 

<br>

|Mantis-ML v2.0 Publication |
| :---- |
|Phenome-wide target identification and validation leveraging Knowledge Graphs, Graph Neural Networks and UK Biobank PheWAS |
|**(to be updated)**|



<br>

Installation
============


**Requirements:** `Python3.8`

It is highly recommended to **create a new virtual environment** (e.g. with `conda`) before installing `mantis-ml-2`:
```
conda create -n mantis_ml_2 python=3.8          # requires Python 3.8
conda activate mantis_ml_2			# activate the newly created conda environment
```

`mantis-ml-2` can then be installed through the github repository:

```
git clone THIS_REPO
cd THIS_REPO
```

`mantis-ml-2` can then be installed through the following command:

```
pip install -e .
```

<br>


---

<br>


You may now call the following script from the command line:
- **`mantisml2`**: run the command with `-h` to see all available options.


<br><br>



Run
===


#### Required field:
- `Disease name (-d)`: (*free text*), e.g. `"diabetes mellitus"`, `"asthma"`
- `Output folder (-o)`: **location of output files, including gene-disease predictions**


You may find the full list of options here**link to paste `-h` at later section**.



<br>

### Example

```
mantisml2 -d "diabetes mellitus" -o mantisml_diabetes_out -m sgcn
```

<br>



#### `mantisml2` Output - check which folders are ouputted and highlight most important ones
`mantisml2` predictions for all genes and across all classifiers can be found at **`[output_dir]/Gene-Predictions`**. 
<br>
The `AUC_performance_by_Classifier.pdf` file under the same dir contains information about the AUC performance per classifier and thus informs about the best performing classifier.

Output figures from all steps during the `mantis-ml-2` run (e.g. *Exploratory Data Analysis/EDA, supervised-learning, unsupervised-learning*) can be found under **`[output_dir]/Output-Figures`**.

<br>



`mantisml2`
=========


```
arguments:
  -h, --help            show this help message and exit
  -d DISEASE_NAME       Disease name. [Required]

  -t FEATURES_TABLE     Extra features table to merge with

  -l NULL_IMP           Null imputation method if external features table is given. Can be either zero, or median, or the path to a tsv file specifying the imputation method for each column separately

  -e EXACT_TERMS        Terms to match against using regular expression matching

  -z FUZZY_TERMS        Terms to match against using NLP

  -x EXCLUDE_TERMS      Terms to exclude

  -o OUTPUT_DIR         Output directory name
                        (absolute/relative path e.g. ./CKD, /tmp/Epilepsy-testing, etc.)
                        If it doesn't exist it will automatically be created [Required]

  -r {all,pre,boruta,pu,post,post_unsup}
                        Specify type of analysis to run (default: all)

  -f, --fast            Fast training using only 4 classifiers: Extra Trees, Random Forest, SVC and Gradient Boosting.
                        By default, mantis-ml uses 6 supervised models for training: Extra Trees, Random Forest, SVC, Gradient Boosting, XGBoost and Deep Neural Net.

  -m SUPERV_MODELS      Explicitly specify which supervised models to be used for training. This overrides the '-f/--fast' option.
                        - Options:
                         et: Extra Trees
                         rf: Random Forest
                         gb: Gradient Boosting
                         xgb: XGBoost
                         svc: Support Vector Classifier
                         dnn: Deep Neural Net
                         stack: Stacking classifier

                        Multiple models may be specified using a ',' separator, e.g. -m et,rf,stack
                        When this option is not specified, 6 models are trained by default with mantis-ml: Extra Trees, Random Forest, SVC, Gradient Boosting, XGBoost and Deep Neural Net.

  -k KNOWN_GENES_FILE   File with custom list of known genes used for training (new-line separated)

  -n NTHREADS           Number of threads (default: 4)

  -i ITERATIONS         Number of stochastic iterations for semi-supervised learning (default: 10)

  -g SEED_GENES_SOURCE  Resource to extract the seed genes from. either HPO, OT, or GEL (default: HPO)

  --practise_run        Specifies whether to run on a subset of 5 stochastic partitions (default: use all partitions)

  --n_epochs N_EPOCHS   Number of epochs to train the GCN/DNN for (default: 200)

  --learning_rate LEARNING_RATE
                        Learning rate for DNN/GCN (default: 0.01)

  --n_filters N_FILTERS
                        Number of filters for the GCN (default: 16)

  --n_layers N_LAYERS   Number of hidden layers for the GCN or power of adjacency matrix in SGCN (1 recommended for GCN) (default: 2)

  --dropout_ratio DROPOUT_RATIO
                        Dropout ratio for the GCN (default: 0.5)

  --bikg_net BIKG_NET   Network to use with the GCN - InWeb used if none provided otherwise (default: None)

  --no_nlp              Optionally remove NLP and revert to string matching

  --shuffle_features    Can use shuffled features (must be already generated)

  --SGCN_one_max        Whether to ensure the maximum of all adjacency variables is 1 for SGCN (default: False)

  --SGCN_summation      Whether to aggregate by mean or summation (default: mean)

  --SGCN_C SGCN_C       Regularisation parameter for SGCN - smaller the stronger the regularisation (default: 1)

```



### Common installation issues

As `mantis-ml-2` depends on `BioWordVec` embeddings, it can be cleaner to download the embedding file before running
installation, as oppose to the default that downloads the file on the initial run of `mantis-ml-2`.
As such, before running `python setup.py install` we must first download an additional file used as part of the NLP module.
This can be performed through 

```
mkdir mantis_ml/data/bioword2vec_embeddings
wget -O mantis_ml/data/bioword2vec_embeddings/bio_embedding_intrinsic  https://ndownloader.figshare.com/files/12551759
```


Installation may then proceed through running `python setup.py install` in the repo root folder.



