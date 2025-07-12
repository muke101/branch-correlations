import os
import argparse
from pathlib import Path

import m5

from gem5.utils.requires import requires
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.coherence_protocol import CoherenceProtocol
from gem5.isas import ISA
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import Resource,AbstractResource,DiskImageResource,CheckpointResource
from gem5.simulate.simulator import Simulator
from gem5.simulate.exit_event import ExitEvent
from gem5.components.boards.x86_two_disks import TwoDisksX86Board

##########################
# Add and Parse options
##########################

parser = argparse.ArgumentParser(
    description="Configuration"
)

parser.add_argument(
    "--num-cpus",
    type=int,
    help="Num of cores to simulate",
    default=1,
)

parser.add_argument(
    "--fastforward-with-kvm",
    action="store_true",
    default=False,
    help="FastForward using KVM CPU instead",
)

parser.add_argument(
    "--kernel",
    type=str,
    required=True,
    help="Kernel binary to be used",
)

parser.add_argument(
    "--main-disk",
    type=str,
    required=True,
    help="OS Disk",
)

parser.add_argument(
    "--main-disk-partition",
    type=str,
    default="1",
    help="Which partition (main disk) should be loaded",
)

parser.add_argument(
    "--secondary-disk",
    type=str,
    required=True,
    help="Benchmark Disk",
)

parser.add_argument(
    "--secondary-disk-partition",
    type=str,
    default="1",
    help="Which partition (secondary disk) should be loaded",
)

parser.add_argument(
    "--script",
    type=str,
    required=True,
    help="Script to launch the benchmark",
)

parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default="",
    help="Checkpoint directory",
)

parser.add_argument(
    "--checkpoint-num",
    type=int,
    default=0,
    help="Which checkpoint should be loaded 0 1 2 ...",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help="Specific checkpoint to load",
)

parser.add_argument(
    "--exit-on-checkpoint",
    action="store_true",
    default=False,
    help="Exit from the simulation loop when doing a checkpoint.",
)

parser.add_argument(
    "--exit-on-dump-stats",
    action="store_true",
    default=False,
    help="Exit from the simulation loop when doing a dump stats.",
)

parser.add_argument(
    "--exit-on-dump-reset-stats",
    action="store_true",
    default=False,
    help="Exit from the simulation loop when doing a dump reset stats.",
)

parser.add_argument(
    "--exit-on-reset-stats",
    action="store_true",
    default=False,
    help="Exit from the simulation loop when doing a reset stats.",
)

args = parser.parse_args()

##########################
# Requiements to run
##########################

requires(
    isa_required=ISA.X86,
    kvm_required=args.fastforward_with_kvm,
)

##########################
# Disks and Scripts
##########################

kernel=AbstractResource(
    local_path=args.kernel,
)

disk=DiskImageResource(
    local_path=args.main_disk,
    root_partition=args.main_disk_partition,
)
sec_disk=DiskImageResource(
    local_path=args.secondary_disk,
    root_partition=args.secondary_disk_partition,
)
script=args.script

# Find the newest checkpoint
ckpt = None

if args.checkpoint != "":
    ckpt=CheckpointResource(args.checkpoint)
    print(f"Using Checkpoint: {args.checkpoint}");
    
else:
    if args.checkpoint_dir != "":
        
        def checkpoint_filter(d):
            if (d[-1] == 't' and d[-2] == 'p' and d[-3] == 'c' and d[-4] == '.' and d[-5] == '5' and d[-6] == 'm'):
                return True
            return False

        def absoluteFilePaths(directory):
            for dirpath,_,filenames in os.walk(directory):
                for f in filenames:
                    yield os.path.abspath(os.path.join(dirpath, f))

        ckpt_dir=args.checkpoint_dir
        ckpts = list(filter(checkpoint_filter,absoluteFilePaths(ckpt_dir)))
        ckpts.sort()
        
        print(f"Found Checkpoints: {ckpts}")
        
        if (args.checkpoint_num >= len(ckpts)):
            print(f"Checkpoint requested '{args.checkpoint_num}' exceeds the amunt of checkpoints found '{len(ckpts)}'")
            exit(1)
        
        ckpt=CheckpointResource(local_path=ckpts[args.checkpoint_num][:-6])
        print(f"Using Checkpoint: {ckpts[args.checkpoint_num][:-6]}");

##########################
# Architecture
##########################

cache_hierarchy = NoCache()

processor = SimpleProcessor(
    isa=ISA.X86,
    cpu_type=(CPUTypes.KVM if args.fastforward_with_kvm else CPUTypes.ATOMIC),
    num_cores=args.num_cpus,
)

memory = SingleChannelDDR3_1600(size="3GiB")

board = TwoDisksX86Board(
    clk_freq="1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
    secondary_disk=sec_disk,
    root_disk_name="/dev/hda",

    exit_on_dump_stats=args.exit_on_dump_stats,
    exit_on_dump_reset_stats=args.exit_on_dump_reset_stats,
    exit_on_reset_stats=args.exit_on_reset_stats
)

board.set_kernel_disk_workload(
    kernel=kernel,
    disk_image=disk,
    readfile=script,
    checkpoint=ckpt,
)

##########################
# Run the simulator
##########################

def handle_checkpoint():
    while True:
        print("Doing checkpoint")
        checkpoint_dir = Path(m5.options.outdir)
        m5.checkpoint((checkpoint_dir / f"cpt.{str(m5.curTick())}").as_posix())
        yield args.exit_on_checkpoint

simulator = Simulator(
    board=board,
    full_system=True,
    on_exit_event={
        ExitEvent.CHECKPOINT: handle_checkpoint(),
    },
)

simulator.run()

print(
    "Exiting @ tick {} because {}.".format(
        simulator.get_current_tick(), simulator.get_last_exit_event_cause()
    )
)
