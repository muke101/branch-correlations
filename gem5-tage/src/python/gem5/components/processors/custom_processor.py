from m5.util import warn
from .base_cpu_processor import BaseCPUProcessor
from ..processors.custom_core import CustomCore

from .cpu_types import CPUTypes
from ...isas import ISA

from typing import Optional


class CustomProcessor(BaseCPUProcessor):

    def __init__(
        self, cputype, num_cores: int, isa: Optional[ISA] = None
    ) -> None:
        """
        :param cpu_type: The CPU type for each type in the processor.
        :param num_cores: The number of CPU cores in the processor.

        :param isa: The ISA of the processor. This argument is optional. If not
        set the `runtime.get_runtime_isa` is used to determine the ISA at
        runtime. **WARNING**: This functionality is deprecated. It is
        recommended you explicitly set your ISA via SimpleProcessor
        construction.
        """
        if not isa:
            warn(
                "An ISA for the SimpleProcessor was not set. This will "
                "result in usage of `runtime.get_runtime_isa` to obtain the "
                "ISA. This function is deprecated and will be removed in "
                "future releases of gem5. Please explicitly state the ISA "
                "via the processor constructor."
            )
        super().__init__(
            cores=[
                CustomCore(cputype, core_id=i, isa=isa)
                for i in range(num_cores)
            ]
        )
