from m5.params import *
from m5.SimObject import SimObject

class DynBranchTracer(SimObject):
    type = 'DynBranchTracer'
    abstract = True
    cxx_header = "cpu/o3/dynbranchtracer.hh"
    cxx_class = "gem5::o3::DynBranchTracer"
