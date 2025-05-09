"""
Module for loading branch instances from HDF5 datasets created by BranchNet.
Provides instance-by-instance access for inference and explanations.
"""
import h5py
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple
import yaml
from pathlib import Path

@dataclass
class BranchInstance:
    """Metadata about a branch instance's location in HDF5 files"""
    file_path: str
    index: int
    history_start: int
    history_end: int
    workload: str
    checkpoint: int

class BenchmarkBranchLoader:
    """Loads branch instances from HDF5 datasets for a specific benchmark and branch PC"""

    def __init__(self, benchmark: str, branch_pc: str, dataset_type: str = 'test'):
        """
        Initialize loader for a specific benchmark and branch PC.
        
        Args:
            benchmark: Name of benchmark (e.g. '648.exchange2_s')
            branch_pc: Branch PC to load data for as hex string (e.g. '0x1234')
            dataset_type: Type of dataset to load ('test', 'train', or 'validation')
        """
        self.pc_bits = 12  # From BranchNet big.yaml defaults
        self.pc_hash_bits = 12
        self.history_length = 582  # Longest history length from big.yaml
        if not branch_pc.startswith('0x'):
            raise ValueError("branch_pc must be a hex string starting with '0x'")
        try:
            self.branch_pc = int(branch_pc, 16)
        except ValueError:
            raise ValueError(f"Invalid hex format for branch_pc: {branch_pc}")
            
        # File handle management
        self.current_file = None
        self.current_path = None

        self.benchmark = benchmark
        
        # Initialize trace paths and instances
        self.trace_paths = self._get_trace_paths(dataset_type)
        self.instances = self._collect_branch_instances()

    def _get_trace_paths(self, dataset_type: str) -> List[str]:
        """Get all HDF5 file paths for this benchmark's dataset"""
        # Load configs
        repo_root = Path(__file__).parent.parent
        with open(repo_root / 'BranchNet/environment_setup/paths.yaml') as f:
            paths = yaml.safe_load(f)
        with open(repo_root / 'BranchNet/environment_setup/ml_input_partitions.yaml') as f:
            partitions = yaml.safe_load(f)

        # Get dataset inputs for this benchmark
        benchmark_info = partitions[self.benchmark]
        if dataset_type == 'test':
            inputs = benchmark_info['test_set']
        elif dataset_type == 'train':
            inputs = benchmark_info['train_set']
        elif dataset_type == 'validation':
            inputs = benchmark_info['validation_set']
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

        # Build HDF5 file paths
        dataset_dir = Path(paths['ml_datasets_dir']) / self.benchmark
        paths = []
        for input_name in inputs:
            # Each input can have multiple simpoints
            pattern = f"{self.benchmark}.{input_name}.*.hdf5"
            paths.extend(str(p) for p in dataset_dir.glob(pattern))
        return paths

    def _collect_branch_instances(self) -> List[BranchInstance]:
        """Index all instances of this branch across files"""
        instances = []
        for path in self.trace_paths:
            checkpoint = int(path.split(".")[-2])
            workload = path.split(".")[-3]
            if 'train.' in path:
                workload = 'train.'+workload
            if not Path(path).exists():
                print(f"Warning: File not found: {path}")
                continue
            try:
                with h5py.File(path, 'r') as f:
                    dataset_name = f'br_indices_{hex(self.branch_pc)}'
                    if dataset_name in f:
                        indices = f[dataset_name][:]
                        indices = self._remove_incomplete_histories(indices)
                        for idx in indices:
                            if idx >= self.history_length:
                                instances.append(BranchInstance(
                                    file_path=path,
                                    index=idx,
                                    history_start=idx - self.history_length,
                                    history_end=idx + 1,
                                    workload = workload,
                                    checkpoint = checkpoint,
                                ))
            except (OSError, IOError) as e:
                print(f"Warning: Failed to read {path}: {e}")
                continue
            
        if not instances:
            print(f"Warning: No instances found for branch PC {hex(self.branch_pc)}")
        return instances
    
    def _remove_incomplete_histories(self, br_indices):
        """Filters out instances of a branch with incomplete histories.

        Args:
            br_indices: a numpy array of indices of the occurances of a branch
            sorted in increasing order.
            history_length: the minimum expected history length that should be
            available for each instance of the target branch.

        Returns:
            a sorted numpy array of indices, each is guaranteed to meet the history
            length requirement.
        """
        if br_indices.size != 0:
            first_valid_idx = 0
            while (first_valid_idx < len(br_indices) and
                br_indices[first_valid_idx] < self.history_length):
                first_valid_idx += 1
            br_indices = br_indices[first_valid_idx:]

        return br_indices
    
    def __len__(self) -> int:
        """Total number of branch instances"""
        return len(self.instances)

    def get_instance(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single branch instance by index.
        
        Args:
            idx: Index of the instance to retrieve

        Returns:
            Tuple of (preprocessed_history, label)
            - preprocessed_history: numpy array of branch history
            - label: 1 if branch was taken, 0 if not taken
            
        Raises:
            IndexError: If idx is out of range
            IOError: If there is an error reading the HDF5 file
        """
        if not self.instances:
            raise ValueError("No instances available")
        if idx < 0 or idx >= len(self.instances):
            raise IndexError(f"Index {idx} out of range [0, {len(self.instances)})")
            
        instance = self.instances[idx]
        
        # Manage file handle
        if self.current_path != instance.file_path:
            if self.current_file is not None:
                self.current_file.close()
            self.current_file = h5py.File(instance.file_path, 'r')
            self.current_path = instance.file_path

        # Get raw history
        history = self.current_file['history'][
            instance.history_start:instance.history_end]
        
        # Preprocess history (same as original BranchNet)
        processed = self._preprocess_history(history[:-1])
        label = history[-1] & 1
        
        # Convert to torch tensors
        return (torch.from_numpy(processed).long(),
                torch.tensor(label, dtype=torch.long),
                instance.workload, instance.checkpoint)

    def _preprocess_history(self, history: np.ndarray) -> np.ndarray:
        """Apply PC hashing and preprocessing to history"""

        # Copied from BranchNet
        pc_hash_bits = self.pc_hash_bits
        pc_bits = self.pc_bits
        
        pc_mask = (1 << (1 + pc_bits)) - 1
        np.bitwise_and(history, pc_mask, out=history)

        if pc_hash_bits < pc_bits:
            unprocessed_bits = pc_bits - pc_hash_bits
            pc_hash_mask = ((1 << pc_hash_bits) - 1) << 1
            shift_count = 1
            temp = np.empty_like(history)
            while unprocessed_bits > 0:
                np.right_shift(history, shift_count * pc_hash_bits, out=temp)
                np.bitwise_and(temp, pc_hash_mask, out=temp)
                np.bitwise_xor(history, temp, out=history)
                shift_count += 1
                unprocessed_bits -= pc_hash_bits

            stew_mask = (1 << (pc_hash_bits + 1)) - 1
            np.bitwise_and(history, stew_mask, out=history)
            
        return history

    def __del__(self):
        """Cleanup file handle"""
        if self.current_file is not None:
            self.current_file.close()
