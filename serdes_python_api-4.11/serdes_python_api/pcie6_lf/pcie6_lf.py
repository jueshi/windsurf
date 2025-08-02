"""Contains the PCIE6_LF class
"""

from serdes_python_api.shared.shared import Shared

REVISION = '$Revision: #5 $'
DATE = '$Date: 2024/05/16 $'


class PCIE6_LF(Shared):
    """Contains pcie6_lf methods related to configuration and evaluation

    :param Shared: Parent class containing product generic methods
    """
    def initialize(self, clock_args: dict | None = None) -> None:
        """Comprehensive configuration procedure

        :param clock_args: Arguments for set_clock, defaults to None
        """

        print('Initializing EVB')

        # initialize clock chip
        if clock_args is not None:
            self.clock_args = clock_args
        self._set_clock()
        self._set_part()

    def get_adaptation_results(self, lane: int) -> dict:
        """Retrieve adaptation results
        """

        print(f'Retrieving adaptation results on lane {lane}')
        request = {'operation': 'function',
                   'function_name': 'pcie6_lf_get_rx_adapt_codes',
                   'param_names': 'lane',
                   'params': lane}
        response = self.talk(request)
        return response

    def get_calibration_results(self, lane: int) -> dict:
        """Retrieve calibration results
        """

        print(f'Retrieving calibration results on lane {lane}')
        request = {'operation': 'function',
                   'function_name': 'pcie6_lf_get_rx_cal_codes',
                   'param_names': 'lane',
                   'params': lane}
        response = self.talk(request)
        return response

    def meas_ber(self, lane: int, reads: int = 1, target: float = 1e-12) -> float:
        """Measure the bit error rate (BER)

        :param reads: Number of reads
        :param target: BER measurement target, defaults to 1e-12
        :return: Average BER over reads. If errors is 0, BER is also 0
        """

        return self.eval(f"pcie6_lf_find_ber({lane}, {reads}, {target},'windowed')")['ber_avg']
