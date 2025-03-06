import csv
import subprocess

import torch
from random import random

TOTAL_TIMESTEPS = 50000
NUM_TRIALS = 50


def run_rewards_parametrization():
    """
    Monte Carlo simulation to find the best reward weights for the PPO agent.
    """

    maps = [
        "maps/16x16/basesWorkers16x16B.xml",
        "maps/16x16/basesWorkers16x16C.xml",
        "maps/16x16/basesWorkers16x16D.xml",
        "maps/16x16/basesWorkers16x16E.xml",
        "maps/16x16/basesWorkers16x16F.xml",
    ]

    with open('./models/lr_coeffs.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trial', 'lr-coeff'])

        for i in range(NUM_TRIALS):
            weights = torch.tensor([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])

            lr_coeff = random() * 10 + 0.1

            writer.writerow([i + 1, lr_coeff])
            # weights = torch.abs(weights)

            # writer.writerow([i + 1] + weights.tolist())

            args = [
                "python",
                "ppo_gridnet.py",
                "--total-timesteps",
                str(TOTAL_TIMESTEPS),
                "--seed", "1",
                "--prod-mode",
                "--wandb-project-name", "microrts-py",
                "--exp-name",
                f"lr-parametrization-{i + 1}",
                "--train-maps",
                *maps,
                "--reward-weight",
                *map(str, weights.tolist()),
                "--lr-coeff",
                str(lr_coeff),
            ]

            print(f"Starting learning rate parametrization {i + 1}")
            print(" ".join(args))
            process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if not process.stdout or not process.stderr:
                print("Error starting process")
                return

            print(f"Using lr-coeff={lr_coeff}")

            while True:
                if process.poll() is not None:
                    break
                output = process.stdout.readline()
                if output:
                    print(output.strip())

            stderr_output = process.stderr.read()
            if stderr_output:
                print(stderr_output.strip())

            process.stdout.close()
            process.stderr.close()
            process.wait()
            print(f"Finished learning rate parametrization {i + 1}")


if __name__ == "__main__":
    run_rewards_parametrization()
