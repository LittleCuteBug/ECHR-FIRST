# ECHR-FIRST

Install necessary packages:
```
!pip install datasets transformers evaluate
```

Run with sampled data to make sure it's runable:\
```
git clone https://github.com/LittleCuteBug/ECHR-FIRST.git
cd echr-first
python longformer_base.py --task_name base --dataset ecthr_a --learning_rate=1e-05 -p saveLongformerModels --seed_number 42 --test
```

Run with task 1
```
python longformer_base.py --task_name base --dataset ecthr_a --learning_rate=1e-05 -p saveLongformerModels --seed_number 42
```

Run with task 2
```
python longformer_base.py --task_name add_true_label --dataset ecthr_a --learning_rate=1e-05 -p saveLongformerModels --seed_number 42
```

Run with task 3
```
python longformer_base.py --task_name add_false_label --dataset ecthr_a --learning_rate=1e-05 -p saveLongformerModels --seed_number 42
```

Arguments:
```
usage: longformer_base.py [-h]
                          -t {base,add_true_label,add_false_label}
                          --dataset {ecthr_a,ecthr_b}
                          [-n NUM_EPOCHS]
                          [-lr LEARNING_RATE]
                          -p SAVING_PATH_ROOT
                          [-s SEED_NUMBER]
                          [--test]
                          [--train_batch_size TRAIN_BATCH_SIZE]
                          [--eval_batch_size EVAL_BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -t {base,add_true_label,add_false_label}, --task_name {base,add_true_label,add_false_label}
  --dataset {ecthr_a,ecthr_b}
  -n NUM_EPOCHS, --num_epochs NUM_EPOCHS
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -p SAVING_PATH_ROOT, --saving_path_root SAVING_PATH_ROOT
  -s SEED_NUMBER, --seed_number SEED_NUMBER
  --test                Run with sampled data for testing
  --train_batch_size TRAIN_BATCH_SIZE
  --eval_batch_size EVAL_BATCH_SIZE

```
