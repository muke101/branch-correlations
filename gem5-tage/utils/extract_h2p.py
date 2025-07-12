import sys
from collections import Counter
import argparse


def analyze_branch_predictions(filename, target_pc, output_file):
    # Store lines matching the PC address
    matching_lines = []

    # Counters
    commit_branch_count = 0
    pattern_counts = Counter(
        {
            "pred:1, taken:0": 0,
            "pred:1, taken:1": 0,
            "pred:0, taken:0": 0,
            "pred:0, taken:1": 0,
        }
    )

    with open(filename) as file:
        for line in file:
            if f"PC:{target_pc}" in line:
                matching_lines.append(line)
                if "Commit branch" in line:
                    commit_branch_count += 1
                for pattern in pattern_counts:
                    if pattern in line:
                        pattern_counts[pattern] += 1

    with open(output_file, 'w') as file:
        file.write(f"Results for PC address {target_pc}:\n")
        file.write(f"  Commit branch count: {commit_branch_count}\n")
        accuracy = (pattern_counts["pred:1, taken:1"] + pattern_counts["pred:0, taken:0"]) / commit_branch_count
        file.write(f"  Accuracy: {accuracy}\n")
        for pattern, count in pattern_counts.items():
            file.write(f"  {pattern}: {count}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract H2P from gem5 log')
    parser.add_argument('log_file', help='Path to the gem5 log file')
    parser.add_argument('pc_address', help='Target PC address')
    parser.add_argument('output_file', help='Path to the output file')
    args = parser.parse_args()
    analyze_branch_predictions(args.log_file, args.pc_address, args.output_file)
