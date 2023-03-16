path_root = "/home/huyqnguyen/Documents/echr-first"
eval_batch_size = 8
for dataset in ["ecthr_a", "ecthr_b"]:
    for learning_rate in [5e-06, 1e-05, 2e-05, 5e-05]:
        for train_batch_size in [1, 2, 4, 8]:
            for seed_number in [43, 97]:
                for task in ["base", "add_true_label", "add_false_label"]:
                    command = (f"python longformer_base.py --task_name {task} --dataset {dataset} " + 
                          f"--learning_rate {learning_rate} " + 
                          f"-p {path_root} " +
                          f"--seed_number {seed_number} " + 
                          f"--train_batch_size {train_batch_size} " + 
                          f"--eval_batch_size {eval_batch_size} ")
                    command = command + "--test " # remove this line
                    print(command)
