"""Contains the PCIE6_V2 class
"""

from serdes_python_api.shared.shared import Shared

REVISION = '$Revision: #5 $'
DATE = '$Date: 2024/03/25 $'


class PCIE6_V2(Shared):
    """Contains pcie6_v2 methods related to configuration and evaluation

    :param Shared: Parent class containing product generic methods
    """

    def initialize(self, clock_args: dict | None = None, pmix_cal: str = '') -> None:
        """Comprehensive configuration procedure

        :param clock_args: Arguments for set_clock, defaults to None
        :param pmix_cal: Filename for custom pmix calibration, must exist in cwd, defaults to ''
        """

        print('Initializing EVB')

        # initialize clock chip
        if clock_args is not None:
            self.clock_args = clock_args
        self._set_clock()
        self._set_part()

        for i, v in enumerate(self.sp_args['rx_en_L']):
            if v:
                self.eval(
                    f"pcie6_v2_demo_adaptation({i}, '{self.sp_args['standard_L']}', '{pmix_cal}')")

    def meas_ber(self, lane: int, reads: int = 1, target: float = 1e-12) -> float:
        """Measure the bit error rate (BER)

        :param reads: Number of reads
        :param target: BER measurement target, defaults to 1e-12
        :return: Average BER over reads. If errors is 0, BER is also 0
        """

        return self.eval(f"pcie6_v2_find_ber({lane}, {reads}, {target},'windowed')")['ber_avg']
