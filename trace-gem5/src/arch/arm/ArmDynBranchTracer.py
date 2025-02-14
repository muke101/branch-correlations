from m5.params import *
from m5.SimObject import SimObject
from m5.objects.DynBranchTracer import DynBranchTracer

class ArmDynBranchTracer(DynBranchTracer):
    type = 'ArmDynBranchTracer'
    cxx_header = "arch/arm/dynbranchtracer.hh"
    cxx_class = "gem5::o3::ArmDynBranchTracer"
