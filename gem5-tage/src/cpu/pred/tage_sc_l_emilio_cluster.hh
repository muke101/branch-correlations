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

typedef __uint128_t uint128_t;
// using AddressColourMap = std::unordered_map<Addr, std::pair<unsigned int, std::string>>;
using AddressColourMap = std::unordered_map<Addr, uint128_t>;

static uint128_t parseUint128(const std::string& str) {
    uint128_t result = 0;
    for (char c : str) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument("Invalid character in uint128_t string");
        }
        result = result * 10 + (c - '0');
    }
    return result;
}

static AddressColourMap readFileContents(const std::string& filename) {
    AddressColourMap addressColourMap;

    std::ifstream file(filename);
    if (!file.is_open()) {
        fatal("Error: Unable to open file with colour labels: %s\n", filename);
    }

    std::string line;
    int lineNum = 0;
    while (std::getline(file, line)) {
        lineNum++;
        std::istringstream iss(line);
        std::string token;

        // Parse the line
        std::vector<std::string> tokens;
        while (iss >> token) {
            tokens.push_back(token);
        }

        if (tokens.size() < 2) {
            fatal("Error: Invalid line format at line %d: %s\n", lineNum, line);
        }

        try {
            // Extract the instruction address
            Addr address = std::stoull(tokens[0]);
            uint128_t colourNumber = parseUint128(tokens[1]);
            addressColourMap[address] = colourNumber;
        } catch (const std::exception& e) {
            fatal("Error parsing line %d: %s\nError: %s\n", lineNum, line, e.what());
        }
    }

    return addressColourMap;
}

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
    bool lookup(ThreadID tid, Addr pc, void* &bp_history) override;
    void updateHistories(ThreadID tid, Addr pc, bool uncond, bool taken,
                         Addr target,  void * &bp_history) override;
    void update(ThreadID tid, Addr pc, bool taken,
                void * &bp_history, bool squashed,
                const StaticInstPtr & inst, Addr target) override;
    virtual void squash(ThreadID tid, void * &bp_history) override;

    static AddressColourMap globalMap;
    static AddressColourMap h2pMap;
    std::map<Addr, std::pair<uint64_t, uint64_t>> h2p_accuracies; //num predictions, num incorrect

    void print_h2p_accuracies() {
        for (const auto &h2p : h2p_accuracies) {
            cprintf("PC: %lx, Total: %lu, Incorrect: %lu\n",
                    h2p.first, h2p.second.first, h2p.second.second);
        }
    }
};

} // namespace branch_prediction
} // namespace gem5

#endif // __CPU_PRED_TAGE_EMILIO_CLUSTER_HH__
