import re
import pandas as pd
import os

original_data = data = {
    "PnPCounterToCab": {"Human-50": 0.02, "Generated-3000": 0.18},
    "PnPCabToCounter": {"Human-50": 0.06, "Generated-3000": 0.28},
    "PnPCounterToMicrowave": {"Human-50": 0.02, "Generated-3000": 0.18},
    "PnPCounterToSink": {"Human-50": 0.02, "Generated-3000": 0.44},
    "PnPCounterToStove": {"Human-50": 0.02, "Generated-3000": 0.06},
    "PnPMicrowaveToCounter": {"Human-50": 0.02, "Generated-3000": 0.08},
    "PnPSinkToCounter": {"Human-50": 0.08, "Generated-3000": 0.42},
    "PnPStoveToCounter": {"Human-50": 0.06, "Generated-3000": 0.28},
    "OpenSingleDoor": {"Human-50": 0.46, "Generated-3000": 0.50},
    "OpenDoubleDoor": {"Human-50": 0.28, "Generated-3000": 0.48},
    "CloseDoubleDoor": {"Human-50": 0.28, "Generated-3000": 0.46},
    "CloseSingleDoor": {"Human-50": 0.56, "Generated-3000": 0.94},
    "OpenDrawer": {"Human-50": 0.42, "Generated-3000": 0.74},
    "CloseDrawer": {"Human-50": 0.80, "Generated-3000": 0.96},
    "TurnOnStove": {"Human-50": 0.32, "Generated-3000": 0.46},
    "TurnOffStove": {"Human-50": 0.04, "Generated-3000": 0.24},
    "TurnOnSinkFaucet": {"Human-50": 0.38, "Generated-3000": 0.34},
    "TurnOffSinkFaucet": {"Human-50": 0.50, "Generated-3000": 0.72},
    "TurnSinkSpout": {"Human-50": 0.54, "Generated-3000": 0.96},
    "CoffeePressButton": {"Human-50": 0.48, "Generated-3000": 0.74},
    "TurnOnMicrowave": {"Human-50": 0.62, "Generated-3000": 0.90},
    "TurnOffMicrowave": {"Human-50": 0.70, "Generated-3000": 0.60},
    "CoffeeServeMug": {"Human-50": 0.22, "Generated-3000": 0.34},
    "CoffeeSetupMug": {"Human-50": 0.00, "Generated-3000": 0.12},
    "ArrangeVegetables": {"FromScratch": 2.00, "Fine-tuned": 12.00},
    "MicrowaveThawing": {"FromScratch": 0.00, "Fine-tuned": 2.00},
    "PrepareCoffee": {"FromScratch": 0.00, "Fine-tuned": 6.00},
    "PreSoakPan": {"FromScratch": 0.00, "Fine-tuned": 4.00},
    "RestockPantry": {"FromScratch": 0.00, "Fine-tuned": 0.00},
    "Average": {"Human-50": 0.288, "Generated-3000": 0.476}
}

# Path to your log file directory
mark = "FromScratch"
# directory_path = '/Users/gaominquan/Downloads/3k-split-epoch/3k-split-epoch'
# directory_path = '/Users/gaominquan/Downloads/3k-split-50/3k-split-50'
# directory_path = '/Users/gaominquan/Downloads/special-3k/special-3k-all-epoches'
# directory_path = '/Users/gaominquan/Downloads/50-epochs-all-2/50-epochs-all-2'
directory_path = '/Users/gaominquan/Downloads/composite-tasks/composite-from-scratch/composite-logs'

# Initialize variables to store parsed data
env_data = {}

# Regular expression to match environment name and success rate
env_pattern = re.compile(r"Env:\s+(\w+)")
success_rate_pattern = re.compile(r'"Success_Rate":\s+([0-9.]+)')

# Track all filenames
filenames = [f for f in os.listdir(directory_path) if f.endswith('.log')]

# Abbreviate filenames for column headers
abbreviated_filenames = {f: f.split("-")[-1].replace(".log", "") for f in filenames}

# Read each log file and parse it
for filename in filenames:
    file_path = os.path.join(directory_path, filename)
    with open(file_path, 'r') as file:
        current_env = None
        for line in file:
            # Check if the line contains an environment name
            env_match = env_pattern.search(line)
            if env_match:
                current_env = env_match.group(1)
                # Ensure the environment key exists in the dictionary
                if current_env not in env_data:
                    env_data[current_env] = {f: [] for f in filenames}

            # Check if the line contains a success rate
            success_rate_match = success_rate_pattern.search(line)
            if success_rate_match and current_env:
                success_rate = float(success_rate_match.group(1))
                # Append the success rate to the corresponding environment and filename list
                env_data[current_env][filename].append(success_rate)

# Calculate mean and standard deviation for each file's success rates
env_stats = {
    "Environment": [],
}

# Add each file’s data as separate columns with abbreviated names
for filename in filenames:
    abbrev_name = abbreviated_filenames[filename]
    env_stats[f"{abbrev_name}_Mean"] = []
    env_stats[f"{abbrev_name}_STD"] = []

# Track the best result and calculate increase percentages
env_stats["Best File"] = []
env_stats["Best File Value"] = []
env_stats["Original Value"] = []
env_stats["Absolute Increase"] = []
env_stats["Relative Increase (%)"] = []

# Compute stats and find the best file for each environment
for env, files in env_data.items():
    env_stats["Environment"].append(env)
    original_value = original_data.get(env, {}).get(mark, None)
    best_file = None
    best_mean = -1
    best_value = None

    for filename in filenames:
        abbrev_name = abbreviated_filenames[filename]
        rates = files.get(filename, [])
        if rates:
            # Calculate mean and standard deviation for the top two rates
            top_two_rates = sorted(rates, reverse=True)[:1]
            mean_rate = round(sum(top_two_rates) / len(top_two_rates), 2)
            std_rate = round(pd.Series(top_two_rates).std(), 2)

            # Append stats for each file
            env_stats[f"{abbrev_name}_Mean"].append(mean_rate)
            env_stats[f"{abbrev_name}_STD"].append(std_rate)

            # Determine if this file has the best mean rate for this environment
            if mean_rate > best_mean:
                best_mean = mean_rate
                best_file = abbrev_name
                best_value = mean_rate
        else:
            # Append None for missing data
            env_stats[f"{abbrev_name}_Mean"].append(None)
            env_stats[f"{abbrev_name}_STD"].append(None)

    # Calculate absolute and relative increase percentages
    if original_value is not None and best_value is not None:
        absolute_increase = round(best_value - original_value, 2)
        relative_increase = round((absolute_increase / original_value) * 100, 2) if original_value != 0 else None
    else:
        absolute_increase = None
        relative_increase = None

    # Append the best file, its value, and increase percentages for the current environment
    env_stats["Best File"].append(best_file)
    env_stats["Best File Value"].append(best_value)
    env_stats["Original Value"].append(original_value)
    env_stats["Absolute Increase"].append(absolute_increase)
    env_stats["Relative Increase (%)"].append(relative_increase)

# Convert results to a DataFrame and display
if __name__ == "__main__":
    env_stats_df = pd.DataFrame(env_stats)
    print(env_stats_df)
    # env_stats_df.to_csv('/Users/gaominquan/Downloads/new-50-all-cases-2.csv', index=False)
    env_stats_df.to_csv('/Users/gaominquan/Downloads/composite-from-scratch-cases-2.csv', index=False)
