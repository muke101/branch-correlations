/*
 * Copyright (c) 2010, 2016 ARM Limited
 * Copyright (c) 2013 Advanced Micro Devices, Inc.
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
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __CPU_O3_DYN_INST_PTR_HH__
#define __CPU_O3_DYN_INST_PTR_HH__

#include "base/refcnt.hh"
#include <deque>
#include <cstdint>
#include <iostream>
#include <string>

typedef uint64_t InstSeqNum;

namespace gem5
{

namespace o3
{

class DynInst;

using DynInstPtr = RefCountingPtr<DynInst>;
using DynInstConstPtr = RefCountingPtr<const DynInst>;

//no particular reasoning to put this here other than it's needed across O3
/**  History of committed branches */
typedef struct branchInfo {
    bool indirect;
    bool taken;
    uint64_t target;
    InstSeqNum seqNum;
    uint64_t pc;
} branchInfo;

std::ostream& operator<<(std::ostream & os, const branchInfo& b);

/** Rolling branch history. Always pushed at the front, popped at the back.
 *  So, branchHistory[n] = nth oldest branch, branchHistory[0] = newest branch. */
typedef std::deque<branchInfo> BranchHistory;

bool operator==(const BranchHistory a, const BranchHistory b);

//unclear on what exactly this should be, choosing a reasonably high number for now
#define MAX_BRANCH_HISTORY 128


} // namespace o3
} // namespace gem5

#endif // __CPU_O3_DYN_INST_PTR_HH__
