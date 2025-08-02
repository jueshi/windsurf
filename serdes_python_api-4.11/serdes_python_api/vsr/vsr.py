"""Contains the VSR class.
"""

from serdes_python_api.shared.shared import Shared
import numpy as np
import time

REVISION = '$Revision: #8 $'
DATE = '$Date: 2025/01/17 $'


class VSR(Shared):
    """Contains vsr methods.

    :param Shared: Parent class containing product generic methods
    """

    def initialize(self, clock_args: dict | None = None, enable_fec: bool = False, external_clock: bool = False) -> dict:
        """Comprehensive configuration procedure.

        :param clock_args: Arguments for set_clock, defaults to None
        :param enable_fec: Enable FEC
        :param external_clock: External clock module is connected to the board
        """
        print('Initializing EVB')

        # initialize clock chip
        if clock_args is None:
            self.clock_args = {'freqMHz': self.protocol_defaults['refclk']}
        else:
            self.clock_args = clock_args

        if not external_clock:
            self._set_clock()

        self.sp_args = self._fetch_set_part_args()
        
        if enable_fec:
            self.sp_args["PG_PM_mode_L"] = "TC"
        else:
            self.sp_args["PG_PM_mode_L"] = "pmd"

        #self.sp_args["c0_L"] = 42 - self.sp_args["c1_L"] - self.sp_args["cm1_L"] - self.sp_args["cm2_L"]
                
        sp_out = self._set_part()
        return sp_out

    def meas_ber(self, lane: int, reads: int = 1, target: float = 1e-10,meas_mode: str = 'windowed',
                 block: str = 'tc') -> float:
        """Measure the bit error rate (BER).

        :param reads: Number of reads
        :param target: BER measurement target, defaults to 1e-10
        :return: Average BER over reads. If errors is 0, BER is also 0
        """
        return self.eval(f"vsr_find_ber({lane}, {reads}, {target},'windowed','{block}')")['ber_avg']

    
    def set_tx_eq(self, lane: int, tx_eq: list[int]) -> None:
        """Set the TX equalization. Maximum number of taps is 42.

        :param tx_eq: TX equalization settings ([cm2, cm1, c0, c1])
        """

        self.eval(f"vsr_ip_cfg_set_txeq({lane}, {tx_eq}, 'direct_codes', 1)")