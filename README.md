# ECHR-FIRST

Install necessary packages:
```
!pip install datasets transformers evaluate
```

Run with sampled data to make sure it's runable:\
```
git clone https://github.com/LittleCuteBug/ECHR-FIRST.git
cd echr-first
python longformer_base.py --task_name base --learning_rate=1e-05 -p saveLongformerModels --seed 42 --test
```

Run with task 1
```
python longformer_base.py --task_name base --learning_rate=1e-05 -p saveLongformerModels --seed 42
```

Run with task 2
```
python longformer_base.py --task_name add_true_label --learning_rate=1e-05 -p saveLongformerModels --seed 42
```

Run with task 3
```
python longformer_base.py --task_name add_false_label --learning_rate=1e-05 -p saveLongformerModels --seed 42
```
