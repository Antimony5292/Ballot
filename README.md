# Ballot

## Repo Structure

1. `data`: It contains three datasets we used in this paper. 
   
2. `checkpoint`: It contains the models obtained after running the code.

3. `ballot.py`: It contains the source codes of this work and a demo. The way to run the demo cases has been shown [here](#setup).
   
4.  misc: The `README.md` shows how to use our demos, the repo structure, the way to reproduce our experiments and our experiment results. And the `requirement.txt` shows all the dependencies of this work.

```
- data/
- checkpoint/
- ballot.py
- util.py
- README.md
- requirements.txt
```

## <span id="setup">Setup</span>
### (Recommended) Create a virtual environment
Ballot requires specific versions of some Python packages which may conflict with other projects on your system. A virtual environment is strongly recommended to ensure the requirements may be installed safely.

### Install with `conda`
To install the requirements, run:

`conda install -r ./requirements.txt`

Note: The version of Pytorch in this requirements is CPU version. If you need GPU version, please check your CUDA version and [install Pytorch manually](https://pytorch.org/).

### Get datasets
for cifar-100 dataset:

- No additional action is required, pytorch will download the dataset automatically.

for TinyImagenet dataset:
- Download the dataset by running following code in folder `data/tiny-imagenet-200/`
```shell
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

for CelebA dataset:
- Download the following three files from [this site](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):
  
`img_align_celeba.zip`, `list_attr_celeba.txt`, and `list_eval_partition.txt`
- Put them in folder `data/celeba/` and unzip the zip package.

### Run demo
1. First the code needs to be run to get the base training model.

For cifar-100 dataset:
```shell
python ballot.py 
```

For TinyImagenet dataset:
```shell
python ballot.py --dataset tinyimagenet --total_cls 200
```

For CelebA dataset:
```shell
python ballot.py --dataset celeba --total_cls 2
```

2. The timestamp of this trial should appear in folder `checkpoint/`. Copy this timestamp number to run the different methods. Take TinyImagenet as an example, assuming we have gotten the timestamp number `1702450003.6434438` by running the code from the previous step, then:
   
For Ballot pruning:
```shell
python ballot.py --dataset tinyimagenet --total_cls 200 --tiral 1702450003.6434438 --method ballot
```

For standard LTH pruning:
```shell
python ballot.py --dataset tinyimagenet --total_cls 200 --tiral 1702450003.6434438 --method lth
```

For random mask pruning:
```shell
python ballot.py --dataset tinyimagenet --total_cls 200 --tiral 1702450003.6434438 --method randomprune
```