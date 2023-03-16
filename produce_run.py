for dataset in ["ecthr_a", "ecthr_b"]:
    for task in ["base", "add_true_label", "add_false_label"]:
        for learning_rate in [5e-06, 1e-05, 2e-05, 5e-05]:
            for train_batch_size in [1, 2, 4, 8]:
                for seed_number in [43, 97]:
                    path = dataset + '/' + task + "/" + f"lr_{learning_rate}_batch_size_{train_batch_size}_seed_{seed_number}"
                    print(f"python longformer_base.py --task_name {task} --dataset {dataset} " + 
                          f"--learning_rate {learning_rate} " + 
                          f"-p {path} " +
                          f"--seed_number {seed_number} " + 
                          f"--train_batch_size {train_batch_size} " + 
                          f"--eval_batch_size {8} " + 
                          "--test ")
