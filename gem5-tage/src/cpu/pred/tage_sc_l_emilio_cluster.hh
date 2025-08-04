#ifndef __CPU_PRED_TAGE_EMILIO_CLUSTER_HH__
#define __CPU_PRED_TAGE_EMILIO_CLUSTER_HH__

#include <unordered_map>
#include <vector>

#include "base/types.hh"
#include "cpu/pred/bpred_unit.hh"
#include "cpu/pred/tage_base.hh"
#include "cpu/pred/tagescl/tagescl.hpp"
#include "cpu/pred/tagescl_cluster/tagescl.hpp"

#include "params/TAGE_EMILIO_cluster.hh"

namespace gem5
{

namespace branch_prediction
{

class TAGE_EMILIO_cluster: public BPredUnit
{
  private:
    tagescl_cluster::Tage_SC_L<tagescl_cluster::CONFIG_64KB> tage;
    tagescl::Tage_SC_L<tagescl::CONFIG_64KB> tage_baseline;

     /**
     * Fetches the colour of this branch and whether it is hard-to-predict.
     * @param branch_addr The address of the branch to find colour of.
     * @return A pair containing (colour, is_h2p)
     */
    inline std::pair<uint128_t, unsigned> getColour(Addr &branch_addr);

    bool using_correlations = false;

  protected:
    virtual bool predict(ThreadID tid, Addr branch_pc, bool cond_branch,
                         void* &b);

    struct TageEmilioBranchInfo
    {
        uint32_t id;
        uint32_t id_baseline;
        Addr pc;
        tagescl_cluster::Branch_Type br_type;
        bool tage_cluster_prediction;
        bool tage_baseline_prediction;
        unsigned int is_h2p;
        TageEmilioBranchInfo()
        {}
    };

  public:

    TAGE_EMILIO_cluster(const TAGE_EMILIO_clusterParams &params);

    // Base class methods.
    bool lookup(ThreadID tid, Addr pc, void* &bp_history) override;
    void updateHistories(ThreadID tid, Addr pc, bool uncond, bool taken,
                         Addr target,  void * &bp_history) override;
    void update(ThreadID tid, Addr pc, bool taken,
                void * &bp_history, bool squashed,
                const StaticInstPtr & inst, Addr target) override;
    virtual void squash(ThreadID tid, void * &bp_history) override;

    static AddressColourMap globalMap;
    static AddressColourMap h2pMap;
};

} // namespace branch_prediction
} // namespace gem5

#endif // __CPU_PRED_TAGE_EMILIO_CLUSTER_HH__
