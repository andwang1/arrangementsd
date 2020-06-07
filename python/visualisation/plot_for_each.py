import os
from diversity import plot_diversity_in_dir
from pheno_circle import plot_pheno_in_dir
from ae_loss_AE import plot_loss_in_dir_AE
from ae_loss_VAE import plot_loss_in_dir_VAE


EXP_FOLDER = "/home/andwang1/airl/balltrajectorysd/results_exp1/repeated_run1"
BASE_NAME = "results_balltrajectorysd_"
variants = [exp_name.split("_")[-1] for exp_name in os.listdir(EXP_FOLDER)]

print(variants)

for variant in variants:
    os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}")
    exp_names = os.listdir()

    is_full_loss = [False] * len(exp_names)

    if variant == "vae":
        for i, name in enumerate(exp_names):
            if "true" in name:
                is_full_loss[i] = True

    for i, exp in enumerate(exp_names):
        pids = os.listdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}")
        for pid in pids:
            full_path = f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}/{pid}"
            print(f"PROCESSING - {full_path}")
            plot_diversity_in_dir(full_path)
            plot_pheno_in_dir(full_path)
            if variant == "vae":
                plot_loss_in_dir_VAE(full_path, is_full_loss[i])
            else:
                plot_loss_in_dir_AE(full_path)
