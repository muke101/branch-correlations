#include "arch/arm/dynbranchtracer.hh"
#include "cpu/o3/dyn_inst.hh"
#include "arch/arm/regs/int.hh"
#include "debug/BranchTrace.hh"

namespace gem5
{

namespace o3
{

ArmDynBranchTracer::ArmDynBranchTracer(const Params& params)
    : DynBranchTracer(params) {
}

void
ArmDynBranchTracer::traceDynBranch(const DynInstPtr& inst) {
     
    if (!inst->staticInst->isCondCtrl())
        return;
    
    std::unique_ptr<PCStateBase> next_pc(inst->pcState().clone());
    inst->staticInst->advancePC(*next_pc);

    std::cerr << "TRACE: ";
    
    // tick
    std::cerr << ::gem5::curTick() << ",";

    // Instruction address
    std::cerr << inst->pcState().instAddr() << ',';

    // Predicted target address
    std::cerr << inst->predPC->instAddr() << ',';

    // Jump address
    std::cerr << next_pc->instAddr() << ',';
    
    if (*next_pc == *inst->predPC) {
        //not mispredicted
        std::cerr << inst->readPredTaken() << ',';
    }
    else {
        //mispredicted
        std::cerr << !(inst->readPredTaken()) << ',';
    }   

    std::cerr << std::endl;
}


} // namespace o3

} // namespace gem5
