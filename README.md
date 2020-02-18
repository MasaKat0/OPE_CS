This is a repository for reproducing the experiments in 'Off-Policy Evaluation and Learning for External Validity under a Covariate Shift'.

Before executing the program, download datasets [satimage](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale), [vehicle](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle.scale), and [pendigits](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits).

## Experiments of Off-Policy Evaluation
```bash
python cs_ope/cs_ope/experiments/experiment_evaluation.py -p 'satimage'
python cs_ope/cs_ope/experiments/experiment_evaluation.py -p 'vehicle'
python cs_ope/cs_ope/experiments/experiment_evaluation.py -p 'pendigits'
```

## Experiments of Off-Policy Learning
```bash
python cs_ope/cs_ope/experiments/experiment_learning.py -p 'satimage'
python cs_ope/cs_ope/experiments/experiment_learning.py -p 'vehicle'
python cs_ope/cs_ope/experiments/experiment_learning.py -p 'pendigits'
```
