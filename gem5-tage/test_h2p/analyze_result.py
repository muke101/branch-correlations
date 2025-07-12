import sys
from collections import Counter


def analyze_branch_predictions(filename, target_pc):
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

    # Output results
    print(f"Results for PC address {target_pc}:")
    print(f"  Commit branch count: {commit_branch_count}")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count}")


if __name__ == "__main__":
    # Example usage
    # Replace 'log.txt' with your filename and '0x4006ac' with the desired PC address
    analyze_branch_predictions(sys.argv[1], sys.argv[2])
