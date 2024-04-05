# Knowledge Graphs Can be Learned with Just Intersection Features
> This is the official codebase for our paper "Knowledge Graphs Can be Learned with Just Intersection Features" ([OpenReview](https://openreview.net/forum?id=)). Should you need to cite our paper, please use the following BibTeX:

```
@inproceedings{le2024intersectionfeatures,
    title={Knowledge Graphs Can be Learned with Just Intersection Features},
    author={Le, Duy and Zhong, Shaochen and Liu, Zirui and Xu, Shuai and Chaudhary, Vipin and Zhou, Kaixiong and Xu, Zhaozhuo},
    booktitle={ArXiv},
    year={2024},
}
```

## Quickstart

### Option 1
We have prepared an example Jupyter Notebook [`intersection_feature_WN18RR_example.ipynb`](./intersection_feature_WN18RR_example.ipynb) that can be run on Google Colab with one click. Please clone the repo into `MyDrive` of you Google Drive, and then run the notebook with Google Colab.

### Option 2
1. Create a Python environment and **install required dependencies** as listed in `requirements.txt`. Then install the `intersection-feature` module and make `checkpoint/` dir:

```
cd OpenKE/
pip install -e .
pip install torch-geometric
pip install datasketch
mkdir checkpoint
```

2. Compile C++ files

```bash
cd openke
bash make.sh
```

3. Quick Start: We provide initial training script to reproduce our result on WN18RR:
```bash
cd OpenKE/
python examples/train_intersection_feature_WN18RR.py
```
