import torch
import yaml
import os
import sys

import lime
from lime.lime_text import LimeTextExplainer
from torch import nn
import time


sys.path.append(os.getcwd())  # which means the script should be run at labelling/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # add the current directory
sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../BranchNet/src/branchnet"
)  # add the parent directory
dir_tests = os.path.dirname(os.path.abspath(__file__)) + "/tests/"
dir_config = (
    os.path.dirname(os.path.abspath(__file__))
    + "/../BranchNet/src/branchnet/configs/big.yaml"
)

from model import BranchNet
from model import BranchNetTrainingPhaseKnobs


def idx_to_string(idx: int) -> str:
    """Convert an integer index to a string of the form '0x759:taken'."""
    raw = format(idx, "b")
    takenness = raw[-1]
    pc_bin = raw[:-1]
    pc_hex = hex(int(pc_bin, 2))
    return "{}:{}".format(pc_hex, "taken" if takenness == "1" else "not_taken")


def tensor_to_string(tensor: torch.Tensor) -> str:
    """Convert a tensor of integers to a string of '0x759:taken 0x759:not_taken ...'."""
    ret = ""
    for x in tensor:
        ret += idx_to_string(x) + " "
    return ret.strip()


def string_to_idx(tok: str) -> int:
    """Convert a string of hex addresses to a tensor of integers.
    The string should be in the format '0x759:taken 0x759:not_taken ...'."""

    addr, takenness = tok.split(":")
    addr_bin = format(int(addr, 16), "b")
    takenness_bit = "1" if takenness == "taken" else "0"
    return int(addr_bin + takenness_bit, 2)


def string_to_tensor(string: str) -> torch.Tensor:
    """Convert a string of hex addresses to a tensor of integers.
    The string should be in the format '0x759:taken 0x759:not_taken ...'."""

    tokens = string.split(" ")
    return torch.tensor([string_to_idx(tok) for tok in tokens], dtype=torch.int64)


class EvalWrapper:
    def __init__(self, model: nn.Module, device) -> None:

        self.model = model
        self.device = device
        model.to('cuda:'+device)
        self.model.eval()

    def _probs(self, input_data) -> torch.Tensor:
        """Get the probabilities of the input data using the model.
        Args:
            input_data (torch.Tensor): Input data of shape (batch_size, input_length).
        """
        with torch.no_grad():
            input_data.to('cuda:'+self.device)
            output = self.model(input_data)
            probs = torch.sigmoid(output).cpu()
        return probs

    def _prob_from_one_string(self, input_string: str) -> torch.Tensor:
        """Get the probabilities of the input string using the model.
        Args:
            input_string (str): Input string of hex addresses in the format '0x759:taken 0x759:not_taken ...'.
        """
        input_tensor = string_to_tensor(input_string)
        return self._probs(input_tensor.unsqueeze(0))

    def probs_from_list_of_strings(self, instances: torch.Tensor) -> torch.Tensor:
        """Get the probabilities of a list of input strings using the model.
        the output is a tensor of shape (batch_size, 2) where the first column is the probability of not taken and the second column is the probability of taken.
        This output format is for matching the expected output of LimeTextExplainer, or the sklearn predict_proba method.
        Args:
            input_strings (list[str]): List of input strings of hex addresses in the format '0x759:taken 0x759:not_taken ...'.
        """
        positive_class_answer = self._probs(instances)
        negative_class_answer = 1 - positive_class_answer
        return (
            torch.stack([negative_class_answer, positive_class_answer])
            .transpose(0, 1)
            .numpy()
        )

    @staticmethod
    def from_checkpoint(checkpoint_path, device, config_path=dir_config) -> "EvalWrapper":
        """Load a model from a BranchNet checkpoint path and config and return an EvalWrapper instance."""

        torch.set_default_device('cuda:'+device)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        training_phase_knobs = BranchNetTrainingPhaseKnobs()
        model = BranchNet(config, training_phase_knobs)
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda:'+device)))
        model.to('cuda:'+device)
        model.eval()
        return EvalWrapper(model, device)


#with open(dir_tests + "example_history.pt", "rb") as f:
#    test_input = torch.load(f)
#with open(dir_tests + "example_label.pt", "rb") as f:
#    test_label = torch.load(f)

"""

# Example usage of EvalWrapper and LimeTextExplainer with a BranchNet model.

eval_wrapper = EvalWrapper.from_checkpoint(
    dir_tests + "base_0x40b412_checkpoint.pt", config_path=dir_config
)


explainer = LimeTextExplainer(
    class_names=["not_taken", "taken"],
    char_level=False,
    split_expression=lambda x: x.split(" "),
    bow=False,
    feature_selection="lasso_path",
    mask_string="0x000:not_taken",  # Mask string for unknown addresses
)
exp = explainer.explain_instance(
    tensor_to_string(test_input),
    eval_wrapper.probs_from_list_of_strings,
    num_features=5,
    num_samples=2000,
)

print(exp.as_list())

"""
