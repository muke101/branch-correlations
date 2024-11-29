from typing import Optional
from ...utils.requires import requires
from .base_cpu_core import BaseCPUCore
from .cpu_types import CPUTypes
from ...isas import ISA
from ...utils.requires import requires
import importlib
import platform


class CustomCore(BaseCPUCore):

    def __init__(
        self, cputype, core_id: int, isa: ISA
    ):
        requires(isa_required=isa)
        isa = isa

        super().__init__(
            core=cputype(),
            isa=isa,
        )
        self.core.cpu_id = core_id

        self._cpu_type = CPUTypes.O3

    def get_type(self) -> CPUTypes:
        return CPUTypes.O3
