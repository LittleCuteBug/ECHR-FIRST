# ECHR-FIRST

Install necessary packages:
```
!pip install datasets transformers evaluate
```

Run with sampled data to make sure it's runable:\
```
git clone https://github.com/LittleCuteBug/ECHR-FIRST.git
cd echr-first
python longformer_base.py --task_name base --learning_rate=1e-05 -p saveLongformerModels --log_file log.log --seed_number 42 --test 
```

Run with task 1
```
python longformer_base.py --task_name base --learning_rate=1e-05 -p saveLongformerModels --log_file log.log --seed_number 42
```

Run with task 2
```
python longformer_base.py --task_name add_true_label --learning_rate=1e-05 -p saveLongformerModels --log_file log.log --seed_number 42
```

Run with task 3
```
python longformer_base.py --task_name add_false_label --learning_rate=1e-05 -p saveLongformerModels --log_file log.log --seed_number 42
```

Arguments:
```
usage: longformer_base.py [-h] -t {base,add_true_label,add_false_label} [-n NUM_EPOCHS] [-lr LEARNING_RATE] -p MODEL_SAVING_PATH [-s SEED_NUMBER] [--test] --log_file LOG_FILE

optional arguments:
  -h, --help            show this help message and exit
  -t {base,add_true_label,add_false_label}, --task_name {base,add_true_label,add_false_label}
  -n NUM_EPOCHS, --num_epochs NUM_EPOCHS
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -p MODEL_SAVING_PATH, --model_saving_path MODEL_SAVING_PATH
  -s SEED_NUMBER, --seed_number SEED_NUMBER
  --test                Run with sampled data for testing
  --log_file LOG_FILE
```
