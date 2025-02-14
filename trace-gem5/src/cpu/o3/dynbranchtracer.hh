#ifndef __DYNBRANCHTRACER_HH__
#define __DYNBRANCHTRACER_HH__

#include <memory>
#include <string>

#include "sim/sim_object.hh"
#include "params/DynBranchTracer.hh"
#include "cpu/o3/dyn_inst_ptr.hh"

namespace gem5
{

namespace o3
{


class DynBranchTracer : public SimObject
{
public:
    using Params = DynBranchTracerParams;
    DynBranchTracer(const DynBranchTracerParams& params); 
    virtual ~DynBranchTracer() = default;
    
    // Print Dynamic Branch trace 
    virtual void traceDynBranch(const DynInstPtr& inst) = 0;
    
};

} // namespace o3

} // namespace gem5

#endif
