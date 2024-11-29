import os
import argparse

import m5

from gem5.utils.requires import requires
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.cachehierarchies.ruby.mesi_three_level_cache_hierarchy import MESIThreeLevelCacheHierarchy
from gem5.components.processors.custom_processor import CustomProcessor
from gem5.coherence_protocol import CoherenceProtocol
from gem5.isas import ISA
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import Resource,AbstractResource,DiskImageResource,CheckpointResource
from gem5.simulate.simulator import Simulator
from gem5.simulate.exit_event import ExitEvent
from gem5.components.boards.x86_two_disks import TwoDisksX86Board

import importlib

##########################
# Add and Parse options
##########################

parser = argparse.ArgumentParser(
    description="Configuration"
)

parser.add_argument(
    "--cpu",
    type=str,
    help="CPU to use",
    default=1,
)

parser.add_argument(
    "--cpu-name",
    type=str,
    help="CPU Python class name (only if using full cpu paths)",
    default="",
)

parser.add_argument(
    "--num-cpus",
    type=int,
    help="Num of cores to simulate",
    default=1,
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
    "--l1i_size",
    type=str,
    default="",
    help="",
)

parser.add_argument(
    "--l1i_assoc",
    type=str,
    default="",
    help="",
)

parser.add_argument(
    "--l1d_size",
    type=str,
    default="",
    help="",
)

parser.add_argument(
    "--l1d_assoc",
    type=str,
    default="",
    help="",
)

parser.add_argument(
    "--l2_size",
    type=str,
    default="",
    help="",
)

parser.add_argument(
    "--l2_assoc",
    type=str,
    default="",
    help="",
)

parser.add_argument(
    "--l3_size",
    type=str,
    default="",
    help="",
)

parser.add_argument(
    "--l3_assoc",
    type=str,
    default="",
    help="",
)

parser.add_argument(
    "--l3_banks",
    type=int,
    default=1,
    help="",
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

parser.add_argument(
    "--warmup-ticks",
    type=int,
    default=0,
    help="Reset stats after the specified amount of ticks.",
)

parser.add_argument(
    "--warmup-insts",
    type=int,
    default=0,
    help="Reset stats after the specified amount of instructions."
)

parser.add_argument(
    "--max-insts",
    type=int,
    default=0,
    help="Limit execution to the specified amount of ticks."
)


args = parser.parse_args()

# Check that both warmups are not enabled at the same time
if ((args.warmup_ticks != 0) and (args.warmup_insts != 0)):
    print("--warmup-ticks and --warmup-insts cannot be enabled both at the same time")
    exit(1)

##########################
# Requiements to run
##########################

requires(
    isa_required=ISA.X86,
    coherence_protocol_required=CoherenceProtocol.MESI_THREE_LEVEL,
    kvm_required=False,
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

cache_hierarchy = MESIThreeLevelCacheHierarchy (
    l1i_size=args.l1i_size,
    l1i_assoc=args.l1i_assoc,
    l1d_size=args.l1d_size,
    l1d_assoc=args.l1d_assoc,
    l2_size=args.l2_size,
    l2_assoc=args.l2_assoc,
    l3_size=args.l3_size,
    l3_assoc=args.l3_assoc,
    num_l3_banks=args.l3_banks,
)

if (".py" not in args.cpu):
    custom_cpu = getattr(importlib.import_module("cores.x86." + args.cpu), args.cpu + "_CPU")
    print(f"Using {custom_cpu} from internal cores")

else:
    module_name = args.cpu[args.cpu.rfind("/")+1:args.cpu.rfind(".py")]
    
    if (args.cpu_name == ""):
        cpu_name = module_name
    else:
        cpu_name = args.cpu_name

    spec = importlib.util.spec_from_file_location(module_name, args.cpu)
    loaded_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_module)
    custom_cpu = getattr(loaded_module, cpu_name + "_CPU")

    print(f"Using {custom_cpu} from {args.cpu}")

processor = CustomProcessor(
    isa=ISA.X86,
    cputype=custom_cpu,
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
    checkpoint=ckpt,
)

##########################
# Run the simulator
##########################

def warmup_done():
    print("Warmup Done @ tick {} because {}.".format(
        simulator.get_current_tick(), simulator.get_last_exit_event_cause()))
    m5.stats.reset()
    print("Running after warmup @ {}.".format(m5.curTick()))
    yield False
    print("We should't come here...")
    yield True

def warmup_done_inst():
    first = 0
    doing_warmup = args.warmup_insts > 0
    while True:
        while doing_warmup:
            if (first == 0):
                print("Warmup Done @ tick {} because {}.".format(
                    simulator.get_current_tick(), simulator.get_last_exit_event_cause()))
                m5.stats.reset()
                print("Running after warmup @ {}.".format(m5.curTick()))
                first = 1
            else:
                # Wi will arrive here once per core
                first = first + 1
                if (first > args.num_cpus):
                    print("MAX_INSTS ticked more than the amount of cores, this should never happen.")
                    yield True
            if (first == args.num_cpus):
                doing_warmup = False
            yield False
        
        if (args.max_insts > 0):
            print("Instruction Limit reached @ tick {}, dumping stats...".format(
                    simulator.get_current_tick()))
            m5.stats.dump()
            print("Done running @ tick {} because {}.".format(
                    simulator.get_current_tick(), simulator.get_last_exit_event_cause()))
            yield True

        yield False

def handle_exit():
    print("Exiting gem5...")
    yield True

simulator = Simulator(
    board=board,
    full_system=True,    
    on_exit_event={
        ExitEvent.SCHEDULED_TICK: warmup_done(),
        ExitEvent.MAX_INSTS: warmup_done_inst(),
        ExitEvent.EXIT : handle_exit(),
    },
)

simulator._instantiate()
print("Starting at Tick {}".format(simulator.get_current_tick()))

# Warmup if needed
if (args.warmup_insts > 0):
    print("Starting warmup for {} insts".format(args.warmup_insts))
    simulator.schedule_max_insts(args.warmup_insts)
elif (args.warmup_ticks > 0):
    print("Starting warmup for {} ticks".format(args.warmup_ticks))
    m5.scheduleTickExitFromCurrent(args.warmup_ticks)
else:
    print("Starting simulation")

if (args.max_insts > 0):
    print("Exit after {} insts".format(args.max_insts))
    simulator.schedule_max_insts(args.max_insts)

simulator.run()

print(
    "Exiting @ tick {} because {}.".format(
        simulator.get_current_tick(), simulator.get_last_exit_event_cause()
    )
)
