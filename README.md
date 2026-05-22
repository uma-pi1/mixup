# Graph Mixup

## Installation

**Prerequisites:** Linux (x86_64), Conda, MySQL

Install conda environment:
```
conda env create -f environment.yml
```

Activate conda environment:
```
conda activate graph_mixup
```

Additionally install `graph_exporter` module:
```
pip install ./graph_exporter
```

Create an new user and an empty database in MySQL:
```
sudo mysql
mysql> CREATE DATABASE graphs;
mysql> CREATE USER 'user' IDENTIFIED BY 'secret';
mysql> GRANT ALL PRIVILEGES on graphs.* TO 'user';
```

Store database credentials in `.env`:
```
# Graph Database:
GED_DB_CONNECTION=mysql
GED_DB_HOST=127.0.0.1
GED_DB_DATABASE=graphs
GED_DB_USER=user
GED_DB_PASSWORD=secret

# HPO Database (better to also use an RDB):
OPTUNA_DB_CONNECTION=sqlite
OPTUNA_DB_HOST=
OPTUNA_DB_DATABASE=optuna.sqlite
OPTUNA_DB_USER=
OPTUNA_DB_PASSWORD=

# Experimental Results Database:
EXPERIMENT_DB_CONNECTION=sqlite
EXPERIMENT_DB_HOST=
EXPERIMENT_DB_DATABASE=experiments.sqlite
EXPERIMENT_DB_USER=
EXPERIMENT_DB_PASSWORD=
```


Store database credentials in `graph_mixup/ged_database/alembic.ini` (line 64):
```
sqlalchemy.url = mysql://user:secret@127.0.0.1/graphs
```

Run database migrations:
```
cd graph_mixup/ged_database/
alembic upgrade head
```

## Most important modules

### `graph_mixup.import_graphs` 

**Purpose:** import vanilla graphs and mixup graphs from (FGW-Mixup and GeoMix)

**Instructions:**
```
usage: __main__.py [-h] [--path PATH] --dataset_name {REDDIT-BINARY,REDDIT-MULTI-5K,IMDB-BINARY,IMDB-MULTI,PROTEINS,COLLAB,MUTAG,ENZYMES,NCI1} [--method_name {if_mixup,g_mixup,fgw_mixup,s_mixup,submix,geomix,ged_mixup}] [--sample_edges] [--verbose]

options:
  -h, --help            show this help message and exit
  --path PATH, -p PATH  path to the root directory of the mixup graphs
  --dataset_name {REDDIT-BINARY,REDDIT-MULTI-5K,IMDB-BINARY,IMDB-MULTI,PROTEINS,COLLAB,MUTAG,ENZYMES,NCI1}
  --method_name {if_mixup,g_mixup,fgw_mixup,s_mixup,submix,geomix,ged_mixup}
  --sample_edges, -se   Sample edges using Bernoulli distribution with their weights.
  --verbose, -v
```

**Example:** Import `MUTAG` dataset:
```
python -m graph_mixup.import_graphs --dataset_name MUTAG
```

### `graph_mixup.__main__`

**Purpose:** conduct experiments with main experimental pipeline including HPO

**Instructions:**
```
usage: __main__.py [-h] --num_trials NUM_TRIALS --study_timeout STUDY_TIMEOUT [--train_timeout TRAIN_TIMEOUT] [--device DEVICE] [--seed SEED] [--cv_seed CV_SEED]
                   [--num_workers NUM_WORKERS] [--log_dir LOG_DIR] [--data_dir DATA_DIR] [--num_test_rounds NUM_TEST_ROUNDS] [--max_epochs MAX_EPOCHS] --patience PATIENCE
                   [--use_baseline] [--use_params_from USE_PARAMS_FROM] --model_name {GCN,GIN} --dataset_name
                   {REDDIT-BINARY,REDDIT-MULTI-5K,IMDB-BINARY,IMDB-MULTI,PROTEINS,COLLAB,MUTAG,ENZYMES,NCI1}
                   [--method_name {emb_mixup,if_mixup,g_mixup,fgw_mixup,s_mixup,submix,geomix,ged_mixup,drop_edge,drop_node,drop_path,perturb_node_attr}]
                   --num_outer_folds NUM_OUTER_FOLDS --num_inner_folds NUM_INNER_FOLDS [--use_inner_holdout] --fold FOLD
                   [--reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS] [--skip_model_selection] [--verbose]

options:
  -h, --help            show this help message and exit
  --num_trials NUM_TRIALS
                        Total number of completed trials (using MaxTrialsCallback).
  --study_timeout STUDY_TIMEOUT
                        Timeout in seconds for the HP optimization of a single fold. Aborts the study even if num_trials is not reached.
  --train_timeout TRAIN_TIMEOUT
                        Timeout in seconds for the training of a single HP configuration.
  --device DEVICE       Device ID
  --seed SEED           Seed that is set in the beginning of each fold's HPO. This affects torch, numpy, optuna, and Python's random module. Note that due to scatter ops
                        on the GPU, results might still differ from one another even though a seed is specified.
  --cv_seed CV_SEED     Seed that is used to determine the dataset folds during cross-validation. If a different value is used, then the dataset is split differently
                        (with high probability).
  --num_workers NUM_WORKERS
                        number of workers in data loaders
  --log_dir LOG_DIR
  --data_dir DATA_DIR   data storage location
  --num_test_rounds NUM_TEST_ROUNDS
                        the number of times a model is initialized, trained, and tested during model assessment
  --max_epochs MAX_EPOCHS
                        number of epochs to train
  --patience PATIENCE   Parameter for early stopping. If the validation accuracy has not improved for the last `patience` validation rounds, training will stop.
  --use_baseline        Uses model HPs from the corresponding baseline experiment.
  --use_params_from USE_PARAMS_FROM
                        Use the best parameters from a prior study of the given name. Will directly proceed to model assessment (i.e., no model selection).
  --model_name {GCN,GIN}
  --dataset_name {REDDIT-BINARY,REDDIT-MULTI-5K,IMDB-BINARY,IMDB-MULTI,PROTEINS,COLLAB,MUTAG,ENZYMES,NCI1}
  --method_name {emb_mixup,if_mixup,g_mixup,fgw_mixup,s_mixup,submix,geomix,ged_mixup,drop_edge,drop_node,drop_path,perturb_node_attr}
                        Choose mixup or augmentation method (can be None).
  --num_outer_folds NUM_OUTER_FOLDS
                        number of folds in cross-validation
  --num_inner_folds NUM_INNER_FOLDS
                        number of inner folds in cross-validation
  --use_inner_holdout   If set, uses inner holdout validation instead of inner cross validation. The validation set size is then given as 1 / num_inner_folds.
  --fold FOLD           outer fold index for current study from [0, num_folds - 1]
  --reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS
  --skip_model_selection
  --verbose, -v
```

**Example:**
```
python -m graph_mixup \
                --num_trials 5 \
                --study_timeout 3600 \
                --num_workers 0 \
                --max_epochs 1000 \
                --patience 20 \
                --num_test_rounds 3 \
                --num_outer_folds 5 \
                --num_inner_folds 5 \
                --device 0 \
                --verbose \
                --model_name GCN \
                --dataset_name MUTAG \
                --fold 0
```

**Notes:**

- The results are available in the `experiments` database (using the database and the credentials you specified in `.env`).
- Before evaluating the baseline, the respective dataset needs to be imported with `graph_mixup.import_graphs`.
- Before evaluating mixup methods:
  - If-Mixup or SubMix: graphs need to be generated using either the respective sub-modules in `graph_mixup.mixup_generation`
  - GeoMix, FGW-Mixup: graphs need to be generated using the authors' modified repos (FGW-Mixup / GeoMix) and then imported using the import module `graph_mixup.import_graphs` while specifying the path to the generated graphs.
    - Install the respective conda environments using the provided `environment.yml` files inside the repositories.
    - Manually install the `graph_exporter` module as before (see _Installation_ above).
    - Example scripts are provided in the respective repos (`generate_graphs.sh`).
  - GED-Mixup: use the module `graph_mixup.ged_mixup`

### `graph_mixup.mixup_generation.*`

**Purpose:** generate mixup items for: If-Mixup, SubMix

**Instructions:**
```
usage: __main__.py [-h] --dataset_name DATASET_NAME [--batch_size BATCH_SIZE] [--max_items_per_pair MAX_ITEMS_PER_PAIR] --max_total MAX_TOTAL [--seed SEED] [--verbose]
                   (--lam LAM | --mixup_alpha MIXUP_ALPHA) [--sample_edges]

options:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
  --batch_size BATCH_SIZE
  --max_items_per_pair MAX_ITEMS_PER_PAIR
                        Maximum number of items per GED mapping. Each item uses a different GED path.
  --max_total MAX_TOTAL
                        Maximum number of mixup items to generate.
  --seed SEED
  --verbose, -v
  --lam LAM             Mixup Lambda
  --mixup_alpha MIXUP_ALPHA
                        Parameter for Beta distribution
  --sample_edges, -se   Sample edges using Bernoulli distribution with their weights.
```

**Example:**
```
python -m graph_mixup.mixup_generation.if_mixup \
                        --dataset_name MUTAG \
                        --batch_size 256 \
                        --mixup_alpha 1.0 \
                        --max_total 16384
```

### `graph_mixup.compute_ged`

**Purpose:** compute GEDs for (i) GED-Mixup and (ii) GED-based analyses

**Instructions:**
```
usage: __main__.py [-h] --dataset_name DATASET_NAME [--n_cpus N_CPUS] [--timeout TIMEOUT] [--lb_threshold LB_THRESHOLD] [--batch_size BATCH_SIZE]
                   [--method_name METHOD_NAME] [--verbose]

options:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
  --n_cpus N_CPUS       Number of CPUs to use
  --timeout TIMEOUT     Timeout in seconds for each GED calculation
  --lb_threshold LB_THRESHOLD
                        Lower bound threshold. If the computed lower bound is above this threshold, GED computation is skipped.
  --batch_size BATCH_SIZE
  --method_name METHOD_NAME
                        Compute GED of mixup graphs with both their parents, and between their parents.
  --verbose, -v
```

**Example:**
```
python -m graph_mixup.compute_ged \
        --dataset_name MUTAG \
        --n_cpus 16 \
        --timeout 20 \
        --lb_threshold 1000 \
        --batch_size 320
```

**Notes:**

- Requires that the dataset has been imported with `graph_mixup.import_graphs`.
- If the provided `ged` binary does not work, you may compile it yourself following 
  the instructions provided [in the respective repo](https://github.com/simon-forb/Graph_Edit_Distance).

### `graph_mixup.ged_mixup`

**Purpose:** generate GED-Mixup graphs

**Instructions:**
```
usage: __main__.py [-h] --dataset_name DATASET_NAME [--batch_size BATCH_SIZE] [--max_items_per_pair MAX_ITEMS_PER_PAIR] --max_total MAX_TOTAL [--seed SEED] [--verbose]
                   (--lam LAM | --mixup_alpha MIXUP_ALPHA) [--max_fail_count MAX_FAIL_COUNT]

options:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
  --batch_size BATCH_SIZE
  --max_items_per_pair MAX_ITEMS_PER_PAIR
                        Maximum number of items per GED mapping. Each item uses a different GED path.
  --max_total MAX_TOTAL
                        Maximum number of mixup items to generate.
  --seed SEED
  --verbose, -v
  --lam LAM             Mixup Lambda
  --mixup_alpha MIXUP_ALPHA
                        Parameter for Beta distribution
  --max_fail_count MAX_FAIL_COUNT, -mfc MAX_FAIL_COUNT
```

**Example:**
```
python -m graph_mixup.ged_mixup \
                        --dataset_name MUTAG \
                        --batch_size 256 \
                        --mixup_alpha 1.0 \
                        --max_items_per_pair 1 \
                        --max_total 16384
```

**Notes:**

- Requires to compute GEDs first for the respective dataset.
