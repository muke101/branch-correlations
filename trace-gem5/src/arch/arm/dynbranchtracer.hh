#ifndef __ARCH_ARM_DYNBRANCHTRACE_HH__
#define __ARCH_ARM_DYNBRANCHTRACE_HH__

#include <vector>
#include <array>

#include "params/ArmDynBranchTracer.hh"
#include "cpu/o3/dynbranchtracer.hh"
#include <unordered_map>
#include <stack>

namespace gem5
{

namespace o3
{



class ArmDynBranchTracer : public DynBranchTracer
{
public:
    using Params = ArmDynBranchTracerParams;
    ArmDynBranchTracer(const Params& params);
    virtual ~ArmDynBranchTracer() = default;

    void traceDynBranch(const DynInstPtr& inst) final;

private: 

    bool m_printed_header{ false };
};

} // namespace o3

} // namespace gem5

#endif

