# Hallucinations in Large Multilingual Translation Models

This repository contains code and data for the "[Hallucinations in Large Multilingual Translation Models](https://arxiv.org/abs/2303.16104)" paper.

## Data with Translations for all setups

You can dowload the translations from all models for all the natural hallucinations setups in the paper by running:

```shell
wget https://web.tecnico.ulisboa.pt/~ist178550/lmt_natural_hallucinations.zip
```

## Generation and Detection of Natural Hallucinations

We will guide through generation with all M2M models and scoring with ALTI+. For more information, please refer to [fairseq documentation](https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100) and the [stopes documentation](https://github.com/facebookresearch/stopes).

---

### Installation

First, start by installing the following dependencies:

```shell
python3 -m venv ltm_env
source ltm_env/bin/activate
pip install -e .
```

Then install fairseq and additional dependencies:

```shell
git clone https://github.com/deep-spin/fairseq-m2m100
cd fairseq-m2m100
git fetch
git checkout alti
pip install -e .
pip install "omegaconf==2.1.2" "hydra-core==1.0.0" "antlr4-python3-runtime==4.8" "sentencepiece==0.1.97" "numpy<=1.23" "pandas<1.5" "fairscale" "einops"
```

After this, download the following files with the sentencepiece models, language dictionaries, etc.:

```shell
wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
wget https://dl.fbaipublicfiles.com/m2m_100/data_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/language_pairs.txt
wget https://dl.fbaipublicfiles.com/m2m_100/language_pairs_small_models.txt 
```

Now, download the models:
```shell
mkdir -p fairseq-m2m100/checkpoints/M2M/
cd fairseq-m2m100/checkpoints/M2M/
wget https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_2_gpus.pt
# rename the model to 12B_last_checkpoint.pt
wget https://dl.fbaipublicfiles.com/m2m_100/418M_last_checkpoint.pt 
wget https://dl.fbaipublicfiles.com/m2m_100/1.2B_last_checkpoint.pt 
```

Download the small-100 model via the Google drive link available in the official small100 repo: https://drive.google.com/file/d/1d6Nl3Pbx7hPUbNkIq-c7KBuC3Vi5rR8D/view?usp=sharing. Unzip it and copy it to the checkpoint folder folder `"fairseq-m2m100/checkpoints/M2M/"` or another folder of your choice. Rename the model ckpt as `"small100_last_checkpoint.pt"`.
___

### Generating Translations

Start by downloading all the data:
```shell
bash download_data.sh
```

🚨 For all the scripts below, make sure to:
* change the `main_path` on the first line of these scripts to your local `llm-hallucination` repo path.
* change the `checkpoint_path` if necessary.

#### English-Centric scenario

We will be running the English-centric scenario for 64 directions (see all directions in the main paper).

To generate the translations, you simply need to run the script:
```shell
$LTM_HALLUCINATIONS_PATH/hallucinations/hallucinations_natural/english_centric/flores/generate_non_english_centric_translations.sh
```

#### Non English-Centric scenario


We will be running the non-English-centric scenario for 25 directions (see all directions in the main paper).

To generate the translations, you simply need to run the script:
```shell
$LTM_HALLUCINATIONS_PATH/hallucinations/hallucinations_natural/non_english_centric/generate_non_english_centric_translations.sh
```

The dataframes with the translations will be saved in `$FAIRSEQ_PATH/data-bin/m2m_non_english_centric/$lang_pair/dataframes`.

#### Specialized Domain Setup

We will be running the specialized domain setup with TICO for 18 directions (see all directions in the main paper).

Just run the following script:
```shell
bash $LTM_HALLUCINATIONS_PATH/hallucinations/hallucinations_natural/specialized_domain/generate_translations.sh
```
___
### Detection with ALTI+
Install our stopes fork:
```shell
cd $LTM_HALLUCINATIONS_PATH
git clone https://github.com/deep-spin/stopes.git
cd stopes
git fetch
git checkout M2M_stopes
pip install -e .
```
Once that is concluded, you can obtain ALTI+ scores by running the following code:
#### English-Centric Setup
```shell
bash $LTM_HALLUCINATIONS_PATH/hallucinations/hallucinations_natural/english-centric/score_with_alti.sh
```

#### Non-English-Centric Setup
```shell
bash $LTM_HALLUCINATIONS_PATH/hallucinations/hallucinations_natural/non_english_centric/score_with_alti.sh
```

#### Specialized Domain Setup
```shell
bash $LTM_HALLUCINATIONS_PATH/hallucinations/hallucinations_natural/specialized_domain/score_with_alti.sh
```
---

## If you found our work/code useful, please cite our work:
```bibtex
@misc{guerreiro2023hallucinations,
      title={Hallucinations in Large Multilingual Translation Models}, 
      author={Nuno M. Guerreiro and Duarte Alves and Jonas Waldendorf and Barry Haddow and Alexandra Birch and Pierre Colombo and André F. T. Martins},
      year={2023},
      eprint={2303.16104},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
