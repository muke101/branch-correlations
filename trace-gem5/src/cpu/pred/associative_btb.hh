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

#ifndef __CPU_PRED_ASSOCIATIVE_BTB_HH__
#define __CPU_PRED_ASSOCIATIVE_BTB_HH__

#include "base/logging.hh"
#include "base/types.hh"
#include "cpu/pred/btb.hh"
#include "mem/cache/replacement_policies/base.hh"
#include "params/AssociativeBTB.hh"

namespace gem5
{

namespace branch_prediction
{

class AssociativeBTB : public BranchTargetBuffer
{
  public:
    AssociativeBTB(const AssociativeBTBParams &params);

    virtual void memInvalidate() override;
    virtual bool valid(ThreadID tid, Addr instPC) override;
    virtual const PCStateBase *lookup(ThreadID tid, Addr instPC,
                           BranchType type = BranchType::NoBranch) override;
    virtual void update(ThreadID tid, Addr instPC,
                        const PCStateBase &target_pc,
                        BranchType type = BranchType::NoBranch,
                        StaticInstPtr inst = nullptr) override;
    const StaticInstPtr getInst(ThreadID tid, Addr instPC) override;

  protected:

    struct BTBEntry : ReplaceableEntry
    {
         /** The entry's tag. */
        Addr tag = 0;

        /** The entry's target. */
        std::unique_ptr<PCStateBase> target;

        /** The entry's thread id. */
        ThreadID tid;

        /** Whether or not the entry is valid. */
        bool valid = false;

        /** Pointer to the static branch instruction at this address */
        StaticInstPtr inst = nullptr;

        /** Saved to detect conflicts */
        Addr pc = 0;
    };


    /** Returns the index into the BTB, based on the branch's PC.
     *  @param inst_PC The branch to look up.
     *  @return Returns the index into the BTB.
     */
    uint64_t getIndex(Addr instPC, ThreadID tid);

    inline Addr getTag(Addr instPC);
    BTBEntry* findEntry(Addr instPC, ThreadID tid);

    /** Internal update call */
    void updateEntry(BTBEntry* &entry, ThreadID tid, Addr instPC,
                    const PCStateBase &target, BranchType type,
                    StaticInstPtr inst);

    /** The actual BTB. */
    std::vector<BTBEntry> btb;
    
    /** Replacement Policy */
    replacement_policy::Base* rp;

    /** The number of entries in the BTB. */
    const unsigned numEntries;

    /** The associativity of the BTB */
    const unsigned assoc;

    /** The number of tag bits per entry. */
    const unsigned tagBits;
    /** Use a tag compression function. */
    const bool compressedTags;

    /** Helper to avoid recomputation for every lookup */
    const uint64_t numSets;
    uint64_t tagShiftAmt;
    uint64_t tagMask;

    /** Number of bits to shift PC when calculating index. */
    const uint64_t instShiftAmt;

    /** Log2 NumThreads used for hashing threadid */
    const unsigned log2NumThreads;

    /** The number of BTB index mask. */
    uint64_t idxMask;

    struct AssociativeBTBStats : public statistics::Group
    {
        AssociativeBTBStats(AssociativeBTB *parent);

        statistics::Scalar conflict;

      statistics::Vector2d replace_map;
      statistics::Vector2d hit_map;
      statistics::Vector replace_vector;
      statistics::Vector hit_vector;

      statistics::Vector replace_way_vector;
      statistics::Vector hit_way_vector;
    } assocStats;

};

} // namespace branch_prediction
} // namespace gem5

#endif // __CPU_PRED_ASSOCIATIVE_BTB_HH__

