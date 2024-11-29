/*
 * Copyright (c) 2024 Eduardo José Gómez Hernández (University of Murcia)
 * Copyright (c) 2022-2023 The University of Edinburgh
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
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
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu/pred/associative_btb.hh"

#include "base/intmath.hh"
#include "base/trace.hh"
#include "debug/BTB.hh"

namespace gem5
{

namespace branch_prediction
{

AssociativeBTB::AssociativeBTB(const AssociativeBTBParams &p)
    : BranchTargetBuffer(p),
        rp(p.replacement_policy),
        numEntries(p.numEntries),
        assoc(p.assoc),
        tagBits(p.tagBits), compressedTags(p.useTagCompression),
        numSets(numEntries/assoc),
        instShiftAmt(p.instShiftAmt),
        log2NumThreads(floorLog2(p.numThreads)),
        assocStats(this)
{
    if (!isPowerOf2(numSets)) {
        fatal("Number of sets is not a power of 2!");
    }

    btb.resize(numEntries);
    for (unsigned i = 0; i < numEntries; ++i) {
        btb[i].valid = false;
        btb[i].setPosition(i / assoc, i % assoc);
        btb[i].replacementData = rp->instantiateEntry();
    }

    idxMask = numSets - 1; 
    tagMask = (1 << tagBits) - 1;
    tagShiftAmt = instShiftAmt + floorLog2(numSets);

    DPRINTF(BTB, "BTB: Creating Associative BTB (entries:%i, assoc:%i, "
                "tagBits:%i/comp:%i, idx mask:%x, numSets:%i, instShiftAmt:%i)\n",
                numEntries, assoc, tagBits, compressedTags, idxMask, numSets, instShiftAmt);
}

void
AssociativeBTB::memInvalidate()
{
    DPRINTF(BTB, "BTB: Invalidate all entries\n");

    for (unsigned i = 0; i < numEntries; ++i) {
        btb[i].valid = false;
        btb[i].setPosition(i / assoc, i % assoc);
        btb[i].replacementData = rp->instantiateEntry();
        rp->reset(btb[i].replacementData);
    }
}

uint64_t
AssociativeBTB::getIndex(Addr instPC, ThreadID tid)
{
    // Need to shift PC over by the word offset.
    return ((instPC >> instShiftAmt)
            ^ (tid << (tagShiftAmt - instShiftAmt - log2NumThreads))
            )
            & idxMask;
}

inline
Addr
AssociativeBTB::getTag(Addr instPC)
{
    return (instPC >> tagShiftAmt) & tagMask;
}

AssociativeBTB::BTBEntry*
AssociativeBTB::findEntry(Addr instPC, ThreadID tid)
{
    unsigned btb_set_idx = getIndex(instPC, tid);
    Addr inst_tag = getTag(instPC);
    DPRINTF(BTB, "BTB::%s: PC:%#x set_idx:%lx inst_tag:%lx\n", __func__, instPC, btb_set_idx, inst_tag);

    assert(((btb_set_idx * assoc) + assoc) <= numEntries);

    for (unsigned i = 0; i < assoc; ++i) {
        unsigned btb_idx = btb_set_idx * assoc + i;
        if (btb[btb_idx].valid
            && inst_tag == btb[btb_idx].tag
            && btb[btb_idx].tid == tid) {
            return &btb[btb_idx];
        }
    }

    return nullptr;
}

bool
AssociativeBTB::valid(ThreadID tid, Addr instPC)
{
    BTBEntry * entry = findEntry(instPC, tid);

    if (entry != nullptr) {
        return true;
    }
    return false;
}

const PCStateBase *
AssociativeBTB::lookup(ThreadID tid, Addr instPC, BranchType type)
{
    stats.lookups[type]++;
    DPRINTF(BTB, "BTB::%s: Looking for entry. PC:%#x\n", __func__, instPC);

    BTBEntry * entry = findEntry(instPC, tid);

    if (entry != nullptr) {
        DPRINTF(BTB, "BTB::%s: Entry found for PC:%#x, real PC:%lx set:%lx way:%lx\n", __func__, instPC, entry->pc, entry->getSet(), entry->getWay());
        // PC is different -> conflict hit.
        if (entry->pc != instPC) {
            assocStats.conflict++;
        }

        rp->touch(entry->replacementData);

        assocStats.hit_map[entry->getSet()][entry->getWay()]++;
        assocStats.hit_vector[entry->getSet()]++;
        assocStats.hit_way_vector[entry->getWay()]++;
        return entry->target.get();
    }
    stats.misses[type]++;
    return nullptr;
}

const StaticInstPtr
AssociativeBTB::getInst(ThreadID tid, Addr instPC)
{
    BTBEntry * entry = findEntry(instPC, tid);

    if (entry != nullptr) {
        return entry->inst;
    }
    return nullptr;
}

void
AssociativeBTB::update(ThreadID tid, Addr instPC,
                    const PCStateBase &target,
                    BranchType type, StaticInstPtr inst)
{
    BTBEntry * entry = findEntry(instPC, tid);

    updateEntry(entry, tid, instPC, target, type, inst);
}

void
AssociativeBTB::updateEntry(BTBEntry* &entry, ThreadID tid, Addr instPC,
                    const PCStateBase &target, BranchType type,
                    StaticInstPtr inst)
{
    if (type != BranchType::NoBranch) {
        stats.updates[type]++;
    }

    if (entry != nullptr) {
        DPRINTF(BTB, "BTB::%s: Updated existing entry. PC:%#x set:%lx way:%lx\n",
                     __func__, instPC, entry->getSet(), entry->getWay());

        rp->touch(entry->replacementData);

        if (entry->pc != instPC)
            assocStats.conflict++;

    } else {
        stats.evictions++;

        std::vector<ReplaceableEntry*> candidates;
        unsigned btb_set_idx = getIndex(instPC, tid);
        for (unsigned i = 0; i < assoc; ++i) {
          unsigned btb_idx = btb_set_idx * assoc + i;
          candidates.push_back(&(btb[btb_idx]));
        }
        entry = dynamic_cast<BTBEntry*>(rp->getVictim(candidates));

        assert(entry != nullptr);
        
        DPRINTF(BTB, "BTB::%s: Replace entry. PC:%#x set:%lx way:%lx\n",
                     __func__, instPC,entry->getSet(), entry->getWay());

        assocStats.replace_map[entry->getSet()][entry->getWay()]++;
        assocStats.replace_vector[entry->getSet()]++;
        assocStats.replace_way_vector[entry->getWay()]++;

        rp->reset(entry->replacementData);
        rp->touch(entry->replacementData);
    }

    entry->tag = getTag(instPC);
    set(entry->target, &target);
    entry->tid = tid;
    entry->valid = true;
    entry->inst = inst;
    entry->pc = instPC;
}


AssociativeBTB::AssociativeBTBStats::AssociativeBTBStats(
                                                AssociativeBTB *parent)
    : statistics::Group(parent),
    ADD_STAT(conflict, statistics::units::Ratio::get(),
             "Number of conflicts. Tag hit but PC different."),
  ADD_STAT(replace_map, statistics::units::Count::get(), "Distribution of replaces"),
  ADD_STAT(hit_map, statistics::units::Count::get(), "Distribution of hits"),
  ADD_STAT(replace_vector, statistics::units::Count::get(), "Distribution of replaces (sets)"),
  ADD_STAT(hit_vector, statistics::units::Count::get(), "Distribution of hits (sets)"),
  ADD_STAT(replace_way_vector, statistics::units::Count::get(), "Distribution of replaces (ways)"),
  ADD_STAT(hit_way_vector, statistics::units::Count::get(), "Distribution of hits (ways)")

{
    using namespace statistics;

  replace_map
    .init(parent->numEntries / parent->assoc, parent->assoc)
    .flags(statistics::none | statistics::oneline);

  hit_map
    .init(parent->numEntries / parent->assoc, parent->assoc)
    .flags(statistics::none | statistics::oneline);

  replace_vector
    .init(parent->numEntries / parent->assoc)
    .flags(statistics::none);

  hit_vector
    .init(parent->numEntries / parent->assoc)
    .flags(statistics::none);

  replace_way_vector
    .init(parent->assoc)
    .flags(statistics::none);

  hit_way_vector
    .init(parent->assoc)
    .flags(statistics::none);

}


} // namespace branch_prediction
} // namespace gem5

