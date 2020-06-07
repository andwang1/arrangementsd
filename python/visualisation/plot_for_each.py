import matplotlib.pyplot as plt
import os
import numpy as np
import pickle as pk
import seaborn as sns
from collections import defaultdict
from diversity import plot_diversity_in_dir
from pheno_circle import plot_pheno_in_dir
from ae_loss_AE import plot_loss_in_dir_AE
from ae_loss_VAE import plot_loss_in_dir_VAE


EXP_FOLDER = "/home/andwang1/airl/balltrajectorysd/results_exp1/test"
BASE_NAME = "results_balltrajectorysd_"
variants = [exp_name.split("_")[-1] for exp_name in os.listdir(EXP_FOLDER)]

# store all data
diversity_dict = defaultdict(list)
loss_dict = defaultdict(list)

for variant in variants:
    os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}")
    exp_names = [exp_name for exp_name in os.listdir() if os.path.isdir(os.path.join(f"{EXP_FOLDER}/{BASE_NAME}{variant}", exp_name))]

    is_full_loss = [False] * len(exp_names)

    if variant == "vae":
        for i, name in enumerate(exp_names):
            if "true" in name:
                is_full_loss[i] = True

    variant_diversity_dict = defaultdict(list)
    variant_loss_dict = defaultdict(list)

    for i, exp in enumerate(exp_names):
        exp_path = f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}"
        pids = [pid for pid in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, pid))]
        for pid in pids:
            full_path = f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}/{pid}"
            print(f"PROCESSING - {full_path}")
            div_dict, max_diversity = plot_diversity_in_dir(full_path)
            variant_diversity_dict[exp].append(div_dict)
            plot_pheno_in_dir(full_path)
            if variant == "vae":
                variant_loss_dict[exp].append(plot_loss_in_dir_VAE(full_path, is_full_loss[i]))
            else:
                variant_loss_dict[exp].append(plot_loss_in_dir_AE(full_path))

        os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}")

        # at experiment level, plot mean and stddev curves for diversity over generations
        x = list(variant_diversity_dict[exp][0].keys()) * len(variant_diversity_dict[exp])
        y = np.array([list(repetition.values()) for repetition in variant_diversity_dict[exp]])
        y = y.flatten()

        sns.lineplot(x, y, estimator="mean", ci="sd", label="Diversity")
        plt.title("Diversity Mean and Std.Dev.")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.hlines(max_diversity, 0, x[-1], linestyles="--", label="Max Diversity")
        plt.legend()
        plt.savefig("diversity.png")
        plt.close()

        # at experiment level, plot losses
        y1 = np.array([repetition["L2"] for repetition in variant_loss_dict[exp]])
        y2 = np.array([repetition["AL"] for repetition in variant_loss_dict[exp]])
        x = list(range(len(y1[0]))) * len(y1)

        sns.lineplot(x, y1.flatten(), estimator="mean", ci="sd", label="Total L2")
        sns.lineplot(x, y2.flatten(), estimator="mean", ci="sd", label="Actual L2")
        plt.title("L2 Mean and Std.Dev.")
        plt.xlabel("Generation")
        plt.ylabel("L2")
        plt.legend()
        plt.savefig("L2.png")
        plt.close()


    # plot variant plots over generations
    os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}")
    generations = list(variant_diversity_dict[exp][0].keys())
    all_experiments = variant_diversity_dict.keys()
    stochasticities = []

    # VAE can have full loss true or false
    if variant != "vae":
        for name in all_experiments:
            components = name.split("_")
            # "random" part of experiment name
            stochasticities.append((components[1][len("random"):]))
    else:
        stochasticities_full = []
        for name in all_experiments:
            components = name.split("_")
            # "random" part of experiment name
            if "fulllosstrue" in name:
                stochasticities_full.append((components[1][len("random"):]))
            else:
                stochasticities.append((components[1][len("random"):]))
        stochasticities_full = sorted(stochasticities_full)

    stochasticities = sorted(stochasticities)

    for generation in generations:
        diversity_values = []
        stochasticity_values = []
        for stochasticity in stochasticities:
            # take correct dictionary according to stochasticity
            components[1] = f"random{stochasticity}"
            for repetition in variant_diversity_dict["_".join(components)]:
                diversity_values.append(repetition[generation])
                stochasticity_values.append(stochasticity)

        sns.lineplot(stochasticity_values, diversity_values, estimator="mean", ci="sd", label="Diversity")
        plt.savefig(f"diversity_gen{generation}.png")
        plt.close()



    # TODO loss plots




    diversity_dict[variant].append(variant_diversity_dict)
    loss_dict[variant].append(variant_loss_dict)

os.chdir(f"{EXP_FOLDER}")

with open("diversity_data.pk", "wb") as f:
    pk.dump(diversity_dict, f)
with open("loss_data.pk", "wb") as f:
    pk.dump(loss_dict, f)