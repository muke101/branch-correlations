import os

base_dir = "/home/bj321/Developer/gem5-new/results"
transformer_dir = "/home/bj321/Developer/branch-transformer-addon/results_single_branch_128_8_3"





def get_best():
  dict = {}
  with open("/home/bj321/Developer/gem5-new/results/full_h2p_baseline/base/m4-emilio/leela.0/results_debug.txt", "r") as f:
    lines = f.readlines()
    groups_of_four = [lines[i:i+5] for i in range(0, len(lines), 5)]
    # print(groups_of_four[0])
    for group in groups_of_four:
      p1t0 = group[0].strip().split(" ")
      p1t1 = group[1].strip().split(" ")
      p0t0 = group[2].strip().split(" ")
      p0t1 = group[3].strip().split(" ")
      acc_line = group[4].strip().split(" ")
      # print(p1t0, p1t1, p0t0, p0t1, acc_line)
      assert p1t0[0] == p1t1[0] == p0t0[0] == p0t1[0] == acc_line[0] and acc_line[1] == "accuracy"
      pc = p1t0[0]
      acc = acc_line[-1]
      mispreds = float(p1t0[-1]) + float(p0t1[-1])
      for dir in os.listdir(transformer_dir):
        if pc in dir:
          with open(os.path.join(transformer_dir, dir, "test_results.txt"), "r") as f:
            for line in f:
              if 'Test Accuracy' in line:
                transformer_acc = line.split(" ")[-1]
                break
      dict[pc] = (acc.strip(), acc.strip(), 0, transformer_acc.strip(), 0, mispreds * 1e-5, 0)


  for dir in os.listdir(base_dir):
    if dir.startswith("full_h2p_brute_force") and os.path.exists(os.path.join(base_dir, dir, "base", "m4", "leela.0", "results_debug.txt")):
      with open(os.path.join(base_dir, dir, "base", "m4", "leela.0", "results_debug.txt"), "r") as f:
        lines = f.readlines()
        groups_of_four = [lines[i:i+5] for i in range(0, len(lines), 5)]
        # print(groups_of_four[0])
        for group in groups_of_four:
          p1t0 = group[0].strip().split(" ")
          p1t1 = group[1].strip().split(" ")
          p0t0 = group[2].strip().split(" ")
          p0t1 = group[3].strip().split(" ")
          acc_line = group[4].strip().split(" ")
          # print(p1t0, p1t1, p0t0, p0t1, acc_line)
          assert p1t0[0] == p1t1[0] == p0t0[0] == p0t1[0] == acc_line[0] and acc_line[1] == "accuracy"
          pc = p1t0[0]
          acc = acc_line[-1]
          mispreds = float(p1t0[-1]) + float(p0t1[-1])
          assert pc in dict, f"PC {pc} not found in dict"
          if dict[pc][1] < acc:
            config = float(dir.split('_')[-1])
            # config = 0
            dict[pc] = (dict[pc][0], acc.strip(), config, dict[pc][3], float(acc.strip()) - float(dict[pc][0]), dict[pc][5], mispreds * 1e-5)
    
  return dict


def get_best_config():
  dict = get_best()
  filtered_dict = {pc: (base_acc, acc, config, transformer_acc, improvement, base_MPKI, MPKI) for pc, (base_acc, acc, config, transformer_acc, improvement, base_MPKI, MPKI) in dict.items() if config >= 0 and improvement > 0.005}
  return [(pc, config) for pc, (base_acc, acc, config, transformer_acc, improvement, base_MPKI, MPKI) in sorted(filtered_dict.items(), key=lambda x: x[1][4], reverse=True)]

if __name__ == "__main__":
  dict = get_best()
  filtered_dict = {pc: (base_acc, acc, config, transformer_acc, improvement, base_MPKI, MPKI) for pc, (base_acc, acc, config, transformer_acc, improvement, base_MPKI, MPKI) in dict.items() if config >= 0 and improvement > 0.005}
  # print(dict)
  # print(filtered_dict)
  total_improvement = 0
  transformer_improvement = 0
  total_MPKI_reduction = 0
  # not_improved = 0
  for pc, (base_acc, acc, config, transformer_acc, improvement, base_MPKI, MPKI) in sorted(filtered_dict.items(), key=lambda x: x[1][4], reverse=True):
    assert improvement == float(acc) - float(base_acc)
    total_improvement += improvement
    transformer_improvement += float(transformer_acc) - float(base_acc)
    total_MPKI_reduction += (base_MPKI - MPKI)
    print(f"PC: {pc}, Base Accuracy: {base_acc}, Accuracy: {acc}, Config: {config}, Transformer Accuracy: {transformer_acc}, Improvement: {improvement}, Base MPKI: {base_MPKI}, MPKI: {MPKI}, MPKI reduction%: {(base_MPKI - MPKI) / base_MPKI * 100}")


  # print(f"average improvement: {total_improvement / len(dict)}")
  print(f"average improvement improved: {total_improvement / (len(filtered_dict))}")
  print(f"average transformer improvement: {transformer_improvement / len(filtered_dict)}")
  print(f"total MPKI reduction: {total_MPKI_reduction}")
  print(f"Number of improved: {len(filtered_dict)}")








