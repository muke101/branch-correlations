/*
 * Copyright (c) 2004-2006 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR#include "mem_dep_unit.hh"
 TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu/o3/phast.hh"
#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "base/compiler.hh"
#include "base/debug.hh"
#include "cpu/o3/dyn_inst.hh"
#include "cpu/o3/inst_queue.hh"
#include "cpu/o3/limits.hh"
#include "dyn_inst_ptr.hh"
#include <cstddef>
#include <iostream>

#include <cmath>

namespace gem5
{

namespace o3
{

PHAST::PHAST(const BaseO3CPUParams &params, MemDepUnit *mem_dep_unit) {

    assert(isPowerOf2(params.phast_num_rows) && "Invalid number of rows per table!\n");

    //TODO: paramertise this with a string and parse it into a list
    historySizes.assign({0, 2, 4, 6, 8, 12, 16, 32});

    depCheckShift = params.LSQDepCheckShift;
    unsigned set_bits = (unsigned)log2((double)(params.phast_num_rows));

    maxBranches = 0;
    selectedTargetBits = 5;
    selectedTargetMask = (1 << selectedTargetBits) - 1;
    memDepUnit = mem_dep_unit;

    unsigned num_tables = historySizes.size();
    paths = std::vector<SimplBlockCache>();
    paths.resize(num_tables, SimplBlockCache());

    for (unsigned i = 0; i < num_tables; ++i) {
        paths[i].init((uint32_t)(set_bits), (uint32_t)(params.phast_associativity), (uint32_t)(params.phast_tag_bits), (uint32_t)(params.phast_max_counter));
    }

}

PHAST::~PHAST()
{
}

void PHAST::init(const BaseO3CPUParams &params, MemDepUnit *mem_dep_unit) {

    assert(isPowerOf2(params.phast_num_rows) && "Invalid number of rows per table!\n");

    //TODO: paramertise this with a string and parse it into a list
    //if you still want power estimation also need to have a non fucked version
    //of my stats counters though, sorry!
    historySizes.assign({0, 2, 4, 6, 8, 12, 16, 32});

    maxBranches = 0;
    selectedTargetBits = 5;
    selectedTargetMask = (1 << selectedTargetBits) - 1;
    memDepUnit = mem_dep_unit;

    depCheckShift = params.LSQDepCheckShift;
    unsigned set_bits = (unsigned)log2((double)(params.phast_num_rows));

    debug = true;

    unsigned num_tables = historySizes.size();
    paths = std::vector<SimplBlockCache>();
    paths.resize(num_tables, SimplBlockCache());

    for (unsigned i = 0; i < num_tables; ++i) {
        paths[i].init((uint32_t)(set_bits), (uint32_t)(params.phast_associativity), (uint32_t)(params.phast_tag_bits), (uint32_t)(params.phast_max_counter));
    }

}

void PHAST::insertStore(Addr store_PC, InstSeqNum store_seq_num, ThreadID id) {
    storeMap[store_PC] = store_seq_num;
}

void PHAST::squash(InstSeqNum squashed_num, ThreadID tid) {
    std::vector<Addr> squashed_pcs;
    for (auto s: storeMap)
        if (s.second >= squashed_num)
            squashed_pcs.push_back(s.first);
    for (auto s: squashed_pcs)
        storeMap.erase(s);
}

PredictionResult PHAST::checkInst(Addr load_pc, InstSeqNum load_seq_num, BranchHistory branchHistory) {

    struct PredictionResult prediction = {0,0,0,0};

    BranchHistory tmp_history;
    for (int i=0; i < branchHistory.size(); i++) {
        tmp_history.push_back(branchHistory[i]);
    }

    if (branchHistory.size() == 0) return prediction;
    unsigned begin = 0;
    while (begin < branchHistory.size() && branchHistory[begin].seqNum > load_seq_num) {
        begin++;
        tmp_history.pop_front();
    }
    if (begin > branchHistory.size()) return prediction; //no +1 branch //TODO: should this be >=?

    if (historySizes[maxBranches] > branchHistory.size()) {
        int i;
        for (i=0; historySizes[i] <= branchHistory.size(); i++);
        maxBranches = i-1;
    }

    bool branch_match;
    bool any_match = false;
    uint64_t hash;
    std::ptrdiff_t distance;
    for (unsigned i = 0; i <= maxBranches && i < historySizes.size(); i++) {
        hash = generateBranchHash(i, historySizes[i], branchHistory, begin);
        //if (i == 4) {
        //    std::cout << "Lookup: " << hex << load_pc << dec << "[" << load_seq_num << "]" << "\n";
        //    paths[i].printBlock(paths[i].getIndex(load_pc, hash));
        //}
        branch_match = false;
        //if (branchMap.find(load_pc) != branchMap.end()) {
        //    for (int i=0; i < branchMap[load_pc].size(); i++) {
        //        if (branchMap[load_pc][i].first == hash && branchMap[load_pc][i].second == tmp_history) {
        //            ++(memDepUnit->stats.matching_history);
        //            branch_match = true;
        //            break;
        //        }
        //    }
        //}
        distance = paths[i].predict(load_pc, hash, branch_match, memDepUnit);
        if (distance) {
            // all paths are read on prediction, so just use that stat to calc reads
            ++(*(memDepUnit->pathWrites[i]));
            prediction.storeQueueDistance = distance;
            prediction.predBranchHistLength = i;
            prediction.predictorHash = hash;
            return prediction;
        }
        //else if (branch_match) ++(memDepUnit->stats.missing_entry);
    }

    return prediction;

    // if (prediction.seqNum && any_match) ++(memDepUnit->stats.hit_with_history);
    // else if (prediction.seqNum) ++(memDepUnit->stats.alias_hit);
}

void PHAST::violation(Addr load_pc, InstSeqNum load_seq_num, InstSeqNum store_seq_num, Addr store_pc, std::ptrdiff_t storeQueueDistance, bool predicted, unsigned predictedPathIndex, uint64_t predictedHash, BranchHistory branchHistory) {

    //corner case of a violation before any branches or no +1 branch
    if (branchHistory.empty() || branchHistory.back().seqNum > store_seq_num) return;

    //taking branch history from commit so first branch is always older than the load
    auto br_it = branchHistory.begin();
    InstSeqNum branch_seq_num;
    do {
        branch_seq_num = br_it->seqNum;
        br_it++;
    } while (br_it != branchHistory.end() && branch_seq_num > store_seq_num);

    unsigned num_branches = (unsigned)std::distance(branchHistory.begin(), br_it);

    //quantise num branches to first lowest path size
    unsigned i;
    if (num_branches >= historySizes.back()) {
        num_branches = historySizes.back();
        i = historySizes.size() - 1;
    }
    else {
        for (i=1; i < historySizes.size(); i++) {
            unsigned size = historySizes[i];
            if (num_branches < size) {
                num_branches = historySizes[i-1];
                --i;
                break;
            }
        }
    }

    if (predicted) {
        paths[predictedPathIndex].updateCommit(load_pc, predictedHash, true);
        ++(memDepUnit->stats.PHASTMispredictions);
        ++(*(memDepUnit->pathReads[predictedPathIndex]));
        ++(*(memDepUnit->pathWrites[predictedPathIndex]));
    }

    uint64_t path_hash = generateBranchHash(i, num_branches, branchHistory, 0);
    paths[i].update(load_pc, path_hash, storeQueueDistance);

    //if (i == 4) {
    //    std::cout << "Update: " << hex << load_pc << dec << " [" << load_seq_num << "]" << " on " << hex << store_pc << dec << " [" << store_seq_num << "]" <<"\n";
    //    paths[i].printBlock(paths[i].getIndex(load_pc, path_hash));
    //}

    //bool exists = false;
    //if (branchMap.find(load_pc) == branchMap.end()) {
    //    branchMap.emplace(load_pc, std::vector<std::pair<uint64_t, std::deque<branchInfo>>>{{path_hash, branchHistory}});
    //}
    //else {
    //    for (int i=0; i < branchMap[load_pc].size(); i++) {
    //        if (branchMap[load_pc][i].first == path_hash && branchMap[load_pc][i].second == branchHistory) {
    //            exists = true;
    //            break;
    //        }
    //    }
    //}
    //if (!exists) {
    //    std::pair<uint64_t, std::deque<branchInfo>> new_entry = {path_hash, branchHistory};
    //    branchMap[load_pc].push_back(new_entry);
    //}

    maxBranches = std::max(maxBranches, i);

    ++(*(memDepUnit->pathReads[i]));
    ++(*(memDepUnit->pathWrites[i]));

}

void PHAST::commit(Addr load_pc, Addr load_addr, unsigned load_size, Addr store_addr, unsigned store_size, unsigned path_index, uint64_t predictor_hash) {

    bool misprediction;
    // if ((store_has_lower_limit && store_has_upper_limit) ||
    //     (store_has_lower_limit && lower_load_has_store_part) ||
    //     (store_has_upper_limit && upper_load_has_store_part) ||
    //     (lower_load_has_store_part && upper_load_has_store_part))
    Addr load_eff_addr1 = load_addr >> depCheckShift;
    Addr load_eff_addr2 = (load_addr + load_size - 1) >> depCheckShift;
    Addr store_eff_addr1 = store_addr >> depCheckShift;
    Addr store_eff_addr2 = (store_addr + store_size - 1) >> depCheckShift;
    if (store_eff_addr2 >= load_eff_addr1 && store_eff_addr1 <= load_eff_addr2)
        misprediction = false;
    else
        misprediction = true;

    if (misprediction) ++(memDepUnit->stats.PHASTMispredictions);
    else ++(memDepUnit->stats.PHASTCorrectPredictions);

    paths[path_index].updateCommit(load_pc, predictor_hash, misprediction);
    ++(*(memDepUnit->pathReads[path_index]));
    ++(*(memDepUnit->pathWrites[path_index]));

}

uint64_t PHAST::generateBranchHash(unsigned path_index, unsigned num_branches, BranchHistory branch_history, unsigned start_indx){
    unsigned end_indx = start_indx + num_branches;
    if (end_indx >= branch_history.size()) return 0; //TODO: shouldn't it be > not >=?
    std::deque<uint64_t> tmp_path;
    tmp_path.clear();
    int bits = 60;
    bitset<BITSETSIZE> h = branch_history[end_indx].target & selectedTargetMask;  // This is the +1 branch.
    tmp_path.push_back(branch_history[end_indx].target);

    unsigned hist_items = 0;
    for (unsigned indx = end_indx-1; indx >= start_indx && hist_items < num_branches; --indx) {
        if (!branch_history[indx].indirect) {
            h <<= 1;
            h[0] = branch_history[indx].taken;
            ++hist_items;
            ++bits;
            tmp_path.push_back(branch_history[indx].taken);
        } else if (selectedTargetMask != 0) {
            h <<= selectedTargetBits;
            h ^= (branch_history[indx].target & selectedTargetMask);
            ++hist_items;
            bits += selectedTargetBits;
            tmp_path.push_back(branch_history[indx].target);
        }
    }

    return foldHistory(h, bits, paths[path_index].getSetBits(), paths[path_index].getTagBits());
}

uint64_t PHAST::foldHistory(bitset<BITSETSIZE> h, int bits, unsigned _setBits, unsigned _tagBits) {
    int width = _setBits + _tagBits;
    bitset<BITSETSIZE> mask((1ULL << width) - 1);
    uint64_t hash = 0;

    while (bits >= width) {
        hash ^= (h & mask).to_ullong();
        h >>= width;
        bits -= width;
    }

    hash ^= (h & bitset<BITSETSIZE>((1ULL << bits) - 1)).to_ullong();

    return hash;
}

void PHAST::clear() {
   maxBranches = 0;

    for (unsigned i = 0; i < paths.size(); ++i) {
        paths[i].clear();
    }

}

int PHAST::SimplBlockCache::init(uint32_t set_bits, uint32_t _associativity, uint32_t tag_bits, uint32_t max_counter_value) {

    tagBits = tag_bits;
    setBits = set_bits;
    associativity = _associativity;
    maxCounterValue = max_counter_value;
    lruCounter = 0;

    cache = std::vector<std::vector<Entry>>();
    cache.resize((1 << setBits), std::vector<Entry>());

    for (uint64_t i = 0; i < (1ULL << setBits); i++) {
        cache[i].resize(associativity, {0,0,0,0});
    }

    //num entries for this path
    return (1 << setBits) * associativity;

}

uint64_t PHAST::SimplBlockCache::xorFold(uint64_t pc, uint64_t history, uint32_t size) const {
    uint64_t mask = (1 << size) - 1;
    uint64_t fold = (history & mask);
    fold = (fold ^ (pc & mask));

    history = (history >> size);

    while (history) {
        fold = (fold ^ (history & mask));
        history = (history >> size);
    }

    return fold;
}

uint64_t PHAST::SimplBlockCache::getIndex(Addr pc, uint64_t history) const {
    pc = (pc ^ (pc >> 2) ^ (pc >> 5));
    uint64_t index = xorFold(0, (pc ^ history), setBits);
    return index;
}
uint64_t PHAST::SimplBlockCache::getTag(Addr pc, uint64_t history) const {
    pc = (pc ^ (pc >> 3) ^ (pc >> 7));
    uint64_t tag = xorFold(0, (pc ^ history), tagBits);
    return tag;
}

PHAST::SimplBlockCache::Entry *PHAST::SimplBlockCache::findEntry(Addr pc, uint64_t history) {
    uint64_t set = getIndex(pc, history);
    uint64_t tag = getTag(pc, history);
    for (uint32_t i = 0; i < associativity; i++) {
        if (cache[set][i].tag == tag) {
            return &(cache[set][i]);
        }
    }
    return nullptr;
}

PHAST::SimplBlockCache::Entry *PHAST::SimplBlockCache::getLRUEntry(uint64_t set) {
    uint32_t lru_way = 0;
    uint64_t lru_value = cache[set][lru_way].lru;
    for (uint32_t i = 0; i < associativity; i++) {
        if (cache[set][i].lru < lru_value) {
            lru_way = i;
            lru_value = cache[set][lru_way].lru;
        }
    }
    return &(cache[set][lru_way]);
}

void PHAST::SimplBlockCache::updateLRU(Entry* entry) {
    entry->lru = lruCounter;
    lruCounter++;
}

std::ptrdiff_t PHAST::SimplBlockCache::predict(Addr pc, uint64_t history, bool branch_match, MemDepUnit *memDepUnit) {
    auto entry = findEntry(pc, history);
    if (branch_match) {
        if (entry == nullptr) ++(memDepUnit->stats.null_entry);
        else if (entry->counter == 0) ++(memDepUnit->stats.counter_is_zero);
        else if (entry->distance == 0) ++(memDepUnit->stats.store_pc_is_zero);
    }

    if (entry == nullptr || entry->counter == 0 || entry->distance == 0) { // no prediction for this PC
        return 0;
    }

    updateLRU(entry);

    return entry->distance;
}

void PHAST::SimplBlockCache::update(Addr pc, uint64_t history, std::ptrdiff_t distance) {
    auto entry = findEntry(pc, history);
    if (entry == nullptr) {
        // no prediction for this entry so far, so allocate one
        entry = getLRUEntry(getIndex(pc, history));
        entry->tag = getTag(pc, history);
        entry->distance = distance;
        entry->counter = maxCounterValue;
        updateLRU(entry);
    } else {
        entry->distance = distance;
        entry->counter = maxCounterValue;
        updateLRU(entry);
    }
}

void PHAST::SimplBlockCache::updateCommit(Addr pc, uint64_t history, bool predictionWrong) {
    auto entry = findEntry(pc, history);
    if (entry == NULL || entry->counter == 0) {
        return;
    }

    if (predictionWrong) {
        if (entry->counter > 0) {
            --entry->counter;
        }
    } else {
        entry->counter = maxCounterValue;
    }

    updateLRU(entry);
}

void PHAST::SimplBlockCache::clear() {

    for (uint64_t i = 0; i < (1ULL << setBits); i++) {
        for (uint32_t j = 0; j < associativity; j++) {
            cache[i][j].tag = 0;
            cache[i][j].distance = 0;
            cache[i][j].lru = 0;
            cache[i][j].counter = 0;
        }
    }
}

void PHAST::SimplBlockCache::printBlock(uint64_t set) {

    for (uint64_t i=0; i < (1ULL << setBits); i++) {
        if (i == set) std::cout << "----> ";
        std::cout << i << ": | ";
        for (uint32_t j=0; j < associativity; j++) {
            std::cout << "[ ";
            std::cout << "Tag: " << cache[i][j].tag << " ";
            std::cout << "SQ D.: " << cache[i][j].distance << " ";
            std::cout << "LRU: " << cache[i][j].lru << " ";
            std::cout << "Cntr: " << cache[i][j].counter << " ";
            std::cout << "] ";
        }
        std::cout << "|\n";
    }

    std::cout << "\n\n";
}

} // namespace o3
} // namespace gem5
