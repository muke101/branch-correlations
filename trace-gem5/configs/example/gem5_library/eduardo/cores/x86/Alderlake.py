from m5.objects import *

class Alderlake_Simple_Int(FUDesc):
    opList = [
        OpDesc(opClass='IntAlu', opLat = 1),
    ]
    count = 1

class Alderlake_Combined(FUDesc):
    opList = [
        OpDesc(opClass='InstPrefetch', opLat = 1),
        OpDesc(opClass='IntAlu', opLat = 1),
        OpDesc(opClass='IntMult', opLat = 4, pipelined = True, issueLat = 2),
        OpDesc(opClass='IntDiv', opLat = 11, pipelined = True, issueLat = 3),
        OpDesc(opClass='IprAccess', opLat = 3, pipelined = True),
        OpDesc(opClass='FloatAdd', opLat = 4, pipelined = True),
        OpDesc(opClass='FloatCmp', opLat = 4, pipelined = True),
        OpDesc(opClass='FloatCvt', opLat = 4, pipelined = True),
        OpDesc(opClass='FloatDiv', opLat = 11, pipelined = True, issueLat = 3),
        OpDesc(opClass='FloatSqrt', opLat = 12, pipelined = True, issueLat = 3),
        OpDesc(opClass='FloatMult', opLat = 4, pipelined = True),
        OpDesc(opClass='FloatMultAcc', opLat = 4, pipelined = True),
        OpDesc(opClass='FloatMisc', opLat = 4, pipelined = True),
        OpDesc(opClass='SimdAdd', opLat = 3, pipelined = True),
        OpDesc(opClass='SimdAddAcc', opLat = 3, pipelined = True),
        OpDesc(opClass='SimdAlu', opLat = 2, pipelined = True),
        OpDesc(opClass='SimdCmp', opLat = 3, pipelined = True),
        OpDesc(opClass='SimdCvt', opLat = 4, pipelined = True),
        OpDesc(opClass='SimdMisc', opLat = 1, pipelined = True),
        OpDesc(opClass='SimdMult', opLat = 4, pipelined = True),
        OpDesc(opClass='SimdMultAcc', opLat = 4, pipelined = True),
        OpDesc(opClass='SimdDiv', opLat = 18, pipelined = True, issueLat = 10),
        OpDesc(opClass='SimdShift', opLat = 1, pipelined = True),
        OpDesc(opClass='SimdShiftAcc', opLat = 1, pipelined = True),
        OpDesc(opClass='SimdSqrt', opLat = 20, pipelined = True, issueLat = 10),
        OpDesc(opClass='SimdReduceAdd', opLat = 5, pipelined = True, issueLat = 2),
        OpDesc(opClass='SimdReduceAlu', opLat = 5, pipelined = True, issueLat = 2),
        OpDesc(opClass='SimdReduceCmp', opLat = 5, pipelined = True, issueLat = 2),
        OpDesc(opClass='SimdFloatAdd', opLat = 4, pipelined = True),
        OpDesc(opClass='SimdFloatAlu', opLat = 4, pipelined = True),
        OpDesc(opClass='SimdFloatCmp', opLat = 4, pipelined = True),
        OpDesc(opClass='SimdFloatCvt', opLat = 4, pipelined = True),
        OpDesc(opClass='SimdFloatMisc', opLat = 1, pipelined = True),
        OpDesc(opClass='SimdFloatMult', opLat = 5, pipelined = True),
        OpDesc(opClass='SimdFloatMultAcc', opLat = 5, pipelined = True),
        OpDesc(opClass='SimdFloatDiv', opLat = 18, pipelined = True, issueLat = 10),
        OpDesc(opClass='SimdFloatSqrt', opLat = 20, pipelined = True, issueLat = 12),
        OpDesc(opClass='SimdFloatReduceAdd', opLat = 5, pipelined = True, issueLat = 2),
        OpDesc(opClass='SimdFloatReduceCmp', opLat = 5, pipelined = True, issueLat = 2),
        OpDesc(opClass='SimdPredAlu', opLat = 5, pipelined = True),
    ]
    count = 3

class Alderlake_Load(FUDesc):
    opList = [
        OpDesc(opClass='MemRead', opLat = 2),
        OpDesc(opClass='FloatMemRead', opLat = 2),
    ]
    count = 3

class Alderlake_Store(FUDesc):
    opList = [
        OpDesc(opClass='MemWrite', opLat = 2),
        OpDesc(opClass='FloatMemWrite', opLat = 2),
    ]
    count = 2

class Alderlake_FUP(FUPool):
    FUList = [
        Alderlake_Simple_Int(),
        Alderlake_Combined(),
        Alderlake_Load(),
        Alderlake_Store(),
    ]

class Alderlake_IndirectBP(SimpleIndirectPredictor):
    indirectSets = 256
    indirectWays = 2
    indirectTagSize = 16
    instShiftAmt = 2

class Alderlake_iTLB(X86TLB):
    size = 64
    entry_type = "instruction"

class Alderlake_dTLB(X86TLB):
    size = 64
    entry_type = "data"

class Alderlake_MMU(X86MMU):
    dtb = Alderlake_dTLB()
    itb = Alderlake_iTLB()

class Alderlake_RAS(ReturnAddrStack):
    numEntries = 64

class Alderlake_BTB(AssociativeBTB):
    useTagCompression = False
    numEntries = 16384
    assoc = 16
    replacement_policy = LRURP()
    instShiftAmt = 0
    tagBits = 63

class Alderlake_BP(TAGE_EMILIO):
    ras = Alderlake_RAS()
    btb = Alderlake_BTB()
    indirectBranchPred = Alderlake_IndirectBP()

class Alderlake_Alderlake(DerivO3Alderlake):
    LQEntries = 192
    SQEntries = 114
    LSQDepCheckShift = 0
    LFSTSize = 1024
    SSITSize = 1024
    decodeToFetchDelay = 1
    renameToFetchDelay = 1
    iewToFetchDelay = 1
    commitToFetchDelay = 1
    renameToDecodeDelay = 1
    iewToDecodeDelay = 1
    commitToDecodeDelay = 1
    iewToRenameDelay = 1
    commitToRenameDelay = 1
    commitToIEWDelay = 1
    fetchWidth = 8
    fetchBufferSize = 16
    fetchToDecodeDelay = 1
    decodeWidth = 6
    decodeToRenameDelay = 1
    renameWidth = 6
    renameToIEWDelay = 1
    issueToExecuteDelay = 1
    dispatchWidth = 12
    issueWidth = 12
    wbWidth = 12
    iewToCommitDelay = 1
    renameToROBDelay = 1
    commitWidth = 8
    squashWidth = 512
    trapLatency = 13
    backComSize = 6
    forwardComSize = 6
    numPhysIntRegs = 332
    numPhysFloatRegs = 332
    numIQEntries = 208
    numROBEntries = 512
    switched_out = False
    fuPool = Alderlake_FUP()
    branchPred = Alderlake_BP()
    mmu = Alderlake_MMU()

