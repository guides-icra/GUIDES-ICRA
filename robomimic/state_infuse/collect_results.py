import re
import pandas as pd
import os

# Path to your log file

directory_path = '/Users/gaominquan/Downloads/infusion_results/50'
#
# file_path = '/Users/gaominquan/Downloads/original-3k-20-more.log'

# Initialize variables to store parsed data
env_data = {}

# Regular expression to match environment name and success rate
env_pattern = re.compile(r"Env:\s+(\w+)")
success_rate_pattern = re.compile(r'"Success_Rate":\s+([0-9.]+)')

# Read the log file and parse it

for filename in os.listdir(directory_path):
    if filename.endswith('.log'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            current_env = None
            for line in file:
                # print(line)
                # Check if the line contains an environment name
                env_match = env_pattern.search(line)
                if env_match:
                    current_env = env_match.group(1)
                    # Ensure the environment key exists in the dictionary
                    if current_env not in env_data:
                        env_data[current_env] = []

                # Check if the line contains a success rate
                success_rate_match = success_rate_pattern.search(line)
                if success_rate_match and current_env:
                    success_rate = float(success_rate_match.group(1))
                    # Append the success rate to the corresponding environment list
                    env_data[current_env].append(success_rate)

# Calculate mean and standard deviation for each environment
env_stats = {
    "Environment": [],
    "Mean Success Rate": [],
    "STD Success Rate": []
}

for env, rates in env_data.items():
    if len(rates) > 0:  # Only process environments with recorded success rates
        top_two_rates = sorted(rates, reverse=True)[:2]
        # Calculate mean and standard deviation for the top two rates
        env_stats["Environment"].append(env)
        env_stats["Mean Success Rate"].append(sum(top_two_rates) / len(top_two_rates))
        env_stats["STD Success Rate"].append(pd.Series(top_two_rates).std())

# Convert results to a DataFrame and display
if __name__ == "__main__":
    env_stats_df = pd.DataFrame(env_stats)
    print(env_stats_df)
    env_stats_df.to_csv('/Users/gaominquan/Downloads/new-50-20-more.csv')
