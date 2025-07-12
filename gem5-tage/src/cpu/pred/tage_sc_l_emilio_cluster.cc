#include "cpu/pred/tage_sc_l_emilio_cluster.hh"

#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/random.hh"
#include "base/trace.hh"
#include "debug/Fetch.hh"
#include "debug/Tage.hh"

namespace gem5
{

namespace branch_prediction
{

// Top 5 most mispredicted branch addresses from HMMER benchmark
std::unordered_set<uint64_t> H2P_BRANCHES = {
  0x40ab34,
  0x40ed24,
  0x40e8e4,
  0x40eaf8,
  0x40e898,
  0x40eb6c,
  0x40ed34,
  0x40a1ac,
  0x40ed44,
  0x40ed14
};
// constexpr uint64_t TOP_BRANCH_ADDRESSES[] = {
//     0x2c5280, // Rank 1
//     0x2adfe4, // Rank 2
//     0x2addfc, // Rank 3
//     0x27e4ec, // Rank 4
//     0x2adea4  // Rank 5
// };
constexpr int NUM_TOP_BRANCHES = 5;

AddressColourMap TAGE_EMILIO_cluster::globalMap = readFileContents("/home/bj321/Developer/branch-transformer-addon/aggregated_relevancy.txt");
AddressColourMap TAGE_EMILIO_cluster::h2pMap = readFileContents("/home/bj321/Developer/branch-transformer-addon/h2p_indexes.txt");
// AddressColourMap TAGE_EMILIO_cluster::globalMap = {};
// AddressColourMap TAGE_EMILIO_cluster::h2pMap = {};

// AddressColourMap TAGE_EMILIO_cluster::globalMap = {{0x40091c, 1}, {0x400964, 1}};
// AddressColourMap TAGE_EMILIO_cluster::h2pMap = {{0x400964, 1}};



TAGE_EMILIO_cluster::TAGE_EMILIO_cluster
  (const TAGE_EMILIO_clusterParams &params) : BPredUnit(params),
  tage(1024), tage_baseline(1024)
{
  // for (const auto& [address, colour] : TAGE_EMILIO_cluster::globalMap) {
  //   DPRINTF(Tage, "Address: %lx, Colour: %d\n", address, colour);
  //   Addr addr = static_cast<Addr>(address);
  //   auto [cluster_id, is_h2p] = getColour(addr);
  //   if (is_h2p) {
  //     std::cout << "Address: " << addr << ", Colour: " << cluster_id << ", Is H2P: " << is_h2p << std::endl;
  //   }
  // }
}

std::pair<uint128_t, unsigned>
TAGE_EMILIO_cluster::getColour(Addr &branch_addr)
{
  // First check if the branch is one of the top 5 most mispredicted branches
  // bool is_h2p = false;
  // for (int i = 0; i < NUM_TOP_BRANCHES; i++) {
  //   if (branch_addr == TOP_BRANCH_ADDRESSES[i]) {
  //     is_h2p = true;
  //     break;
  //   }
  // bool is_h2p = H2P_BRANCHES.count(branch_addr) > 0;
  auto h2p_it = h2pMap.find(branch_addr);
  // is_h2p encodes the index of the h2p, 0 means not h2p, other values are the index of the h2p
  unsigned is_h2p = 0;
  if (h2p_it != h2pMap.end()) {
    is_h2p = h2p_it->second;
  }

  // Then get the cluster color
  auto it = globalMap.find(branch_addr);
  if (it != globalMap.end()) {
    // Return the colour number if found
    uint128_t colour = it->second;
    return std::make_pair(colour, is_h2p);
  } else {
    return std::make_pair(0, is_h2p); // Uncoloured but might be h2p
  }
}

// PREDICTOR UPDATE
void
TAGE_EMILIO_cluster::update(ThreadID tid, Addr pc, bool taken,
              void * &bp_history,
              bool squashed, const StaticInstPtr & inst, Addr target)
{
    TageEmilioBranchInfo *bi = static_cast<TageEmilioBranchInfo*>(bp_history);

    DPRINTF(Tage, "TAGE id:%d update: %lx squashed:%s bp_history:%p\n",
            bi ? bi->id : -1, pc, squashed, bp_history);

    tagescl::Branch_Type baseline_br_type;
    baseline_br_type.is_conditional = bi->br_type.is_conditional;
    baseline_br_type.is_indirect = bi->br_type.is_indirect;

    if (taken == bi->tage_cluster_prediction) {
      stats.tageClusterCorrect++;
    } else {
      stats.tageClusterIncorrect++;
    }
    if (taken == bi->tage_baseline_prediction) {
      stats.tageBaselineCorrect++;
    } else {
      stats.tageBaselineIncorrect++;
    }
    if (bi->tage_baseline_prediction != bi->tage_cluster_prediction) {
      DPRINTF(Tage, "Baseline and cluster made different predictions for %lx, baseline: %d, cluster: %d, taken: %d\n", pc, bi->tage_baseline_prediction, bi->tage_cluster_prediction, taken);
    }

    if (bi->is_h2p) {
      stats.tageH2PPredicted++;
      if (taken == bi->tage_cluster_prediction) {
        stats.tageH2PCorrect++;
      } else {
        stats.tageH2PIncorrect++;
      }
    }

    assert(bp_history);
    if (squashed) {
        // This restores the global history, then update it
        // and recomputes the folded histories.
        tage.flush_branch_and_repair_state(bi->id, pc, bi->br_type, taken,
          target);
        tage_baseline.flush_branch_and_repair_state(bi->id_baseline, pc, baseline_br_type, taken,
          target);
        return;
    }

    tage.commit_state(bi->id, pc, bi->br_type, taken);
    tage_baseline.commit_state(bi->id_baseline, pc, baseline_br_type, taken);
    tage.commit_state_at_retire(bi->id, pc, bi->br_type, taken, target);
    tage_baseline.commit_state_at_retire(bi->id_baseline, pc, baseline_br_type, taken, target);
  
    delete bi;
    bp_history = nullptr;
}

void
TAGE_EMILIO_cluster::squash(ThreadID tid, void * &bp_history)
{
    TageEmilioBranchInfo *bi = static_cast<TageEmilioBranchInfo*>(bp_history);
    DPRINTF(Tage, "TAGE id: %d squash: %lx bp_history:%p\n", bi ? bi->id : -1,
        bi? bi->pc : 0x00, bp_history);
    if (bi) {
      tage.flush_branch(bi->id);
      tage_baseline.flush_branch(bi->id_baseline);
    }
    delete bi;
    bp_history = nullptr;
}

bool
TAGE_EMILIO_cluster::predict(ThreadID tid, Addr pc, bool cond_branch, void* &b,
                    uint32_t cluster_id, bool is_h2p)
{
    uint32_t id = tage.get_new_branch_id();
    uint32_t tage_baseline_id = tage_baseline.get_new_branch_id();
    // Get color and h2p status from our function
    auto [cluster_id_override, is_h2p_override] = getColour(pc);
    
    TageEmilioBranchInfo *bi = new TageEmilioBranchInfo();
    b = (void*)(bi);
    // DPRINTF(Tage, "TAGE id: %d predict: %lx bp_history:%p cluster_id: %d is_h2p: %d\n", 
    //         id, pc, b, cluster_id_override, is_h2p_override);
    bi->id = id;
    bi->id_baseline = tage_baseline_id;
    bi->pc = pc;
    bi->br_type.is_conditional = cond_branch;
    bi->br_type.is_indirect = false;
    bi->is_h2p = is_h2p_override;
    bool tage_cluster_prediction = tage.get_prediction(id, pc, cluster_id, is_h2p);
    bool tage_baseline_prediction = tage_baseline.get_prediction(tage_baseline_id, pc, 0, false);
    bi->tage_cluster_prediction = tage_cluster_prediction;
    bi->tage_baseline_prediction = tage_baseline_prediction;
    return tage_cluster_prediction;
}

bool
TAGE_EMILIO_cluster::lookup(ThreadID tid, Addr pc, void* &bp_history,
                    uint32_t cluster_id, bool is_h2p)
{
    DPRINTF(Tage, "TAGE lookup: %lx %p\n", pc, bp_history);
    bool retval = predict(tid, pc, true, bp_history, cluster_id, is_h2p);

    DPRINTF(Tage, "Lookup branch: %lx; predict:%d; bp_history:%p\n", pc,
    retval, bp_history);

    return retval;
}

bool
TAGE_EMILIO_cluster::lookup(ThreadID tid, Addr pc, void* &bp_history)
{
    return lookup(tid, pc, bp_history, 0, false);
}

void
TAGE_EMILIO_cluster::updateHistories(ThreadID tid, Addr pc, bool uncond,
                         bool taken, Addr target, void * &bp_history)
{
    TageEmilioBranchInfo *bi = static_cast<TageEmilioBranchInfo*>(bp_history);

    DPRINTF(Tage, "TAGE id: %d updateHistories: %lx %p\n", bi ? bi->id : -1,
            pc, bp_history);

    assert(uncond || bp_history);
    if (uncond) {
        DPRINTF(Tage, "UnConditionalBranch: %lx\n", pc);
        predict(tid, pc, false, bp_history, 0, false);
    }
    //bi->br_type.is_conditional = !uncond;

    bi = static_cast<TageEmilioBranchInfo*>(bp_history);
    tagescl::Branch_Type baseline_br_type;
    baseline_br_type.is_conditional = bi->br_type.is_conditional;
    baseline_br_type.is_indirect = bi->br_type.is_indirect;
    // Update the global history for all branches
    tage.update_speculative_state(bi->id, pc, bi->br_type, taken, target);
    tage_baseline.update_speculative_state(bi->id_baseline, pc, baseline_br_type, taken, target);
}

} // namespace branch_prediction
} // namespace gem5
