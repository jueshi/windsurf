"""Contains the PCIE6 class
"""

from serdes_python_api.shared.shared import Shared

REVISION = '$Revision: #5 $'
DATE = '$Date: 2024/09/09 $'


class PCIE6(Shared):
    """Contains pcie6 methods related to configuration and evaluation

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
        ipid = self.eval("get_pid('ip')")
        
        adapt_params = {}
        standard = self.sp_args['standard_L']
        if standard == 'PCIe6_CC' or standard == 'PCIe5_CC':
            adapt_params['std_support_cca'] = 1
        else:
            adapt_params['std_support_cca'] = 0
        
        for i, v in enumerate(self.sp_args['rx_en_L']):
            if v:
                request = {'operation': 'function',
                   'function_name': 'pcie6_adapt_rx',
                   'param_names': ['ipid', 'lane','adapt_params'],
                   'params': [ipid, i, adapt_params]}
                response = self.talk(request)

    def get_adaptation_results(self, lane: int) -> dict:
        """Retrieve adaptation results
        """

        print(f'Retrieving adaptation results on lane {lane}')
        request = {'operation': 'function',
                   'function_name': 'pcie6_get_rx_adapt_codes',
                   'param_names': 'lane',
                   'params': lane}
        response = self.talk(request)
        return response

    def get_calibration_results(self, lane: int) -> dict:
        """Retrieve calibration results
        """

        print(f'Retrieving calibration results on lane {lane}')
        request = {'operation': 'function',
                   'function_name': 'pcie6_get_rx_cal_codes',
                   'param_names': 'lane',
                   'params': lane}
        response = self.talk(request)
        return response

    def meas_ber(self, lane: int, reads: int = 1, target: float = 1e-8) -> float:
        """Measure the bit error rate (BER)

        :param reads: Number of reads
        :param target: BER measurement target, defaults to 1e-12
        :return: Average BER over reads. If errors is 0, BER is also 0
        """

        return self.eval(f"pcie6_find_ber({lane}, {reads}, {target},'windowed')")['ber_avg']
