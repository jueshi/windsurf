"""Contains the e224g class
"""

import csv
import json
from datetime import datetime
from typing import Literal

from serdes_python_api.shared.shared import Shared

REVISION = '$Revision: #8 $'
DATE = '$Date: 2025/01/07 $'


class E224G(Shared):
    """Contains e224g methods related to configuration and evaluation

    :param Shared: Parent class containing product generic methods
    """

    def initialize(self, clock_args: dict | None = None) -> None:
        """Comprehensive configuration procedure

        :param clock_args: arguments for set_clock, defaults to None
        """
        print('Initializing EVB')
        # initialize clock chip
        if clock_args is None:
            self.clock_args = {'freqMHz': self.protocol_defaults['refclk']}
        else:
            self.clock_args = clock_args
        self._set_clock()

        self._set_part()

    def config_clock(self, tx_board:str= 'USB <-> Serial Cable B', rx_board:str= 'SM1B-231027-011 B',
                  tx_clkmod_file: str= 'Si5391_100M_Out0_531p25_OUT_2_9_Registers',
                  rx_clkmod_file: str= 'Si5391_100M_Out0_531p25_OUT_2_9_Registers') -> None:
        self.eval(
            f"e224g_cfg_board_clock('{tx_board}', '{rx_board}', '{tx_clkmod_file}','{rx_clkmod_file}')")

    def select_dongle(self, board:str = 'TX')->None:
        self.eval(
            f"e224g_select_dongle('{board}')")

    def enable_fec(self, tx_lane: int, rx_lane: int, enable: bool, ) -> None:
        """Enables forward error correction
        """

        self.eval(
            f"e224g_enable_fec({tx_lane}, {rx_lane}, {int(enable)})")
        
    def enable_fec_b2b(self, tx_lane: int, rx_lane: int, enable: bool, ) -> None:
        """Enables forward error correction
        """

        self.eval(
            f"e224g_enable_fec_b2b({tx_lane}, {rx_lane}, {int(enable)})")

    def get_rx_args(self) -> dict:
        with open('sp_args_rx.json', encoding='utf-8') as file:
            return json.load(file)

    def get_adaptation_results(self, lane: int) -> dict:
        """Retrieve adaptation results
        """

        print(f'Retrieving adaptation results on lane {lane}')
        request = {'operation': 'function',
                   'function_name': 'e224g_read_adapt_codes',
                   'param_names': 'lane',
                   'params': lane}
        response = self.talk(request)
        return response

    def get_calibration_results(self, lane: int) -> dict:
        """Retrieve calibration results
        """

        print(f'Retrieving calibration results on lane {lane}')
        request = {'operation': 'function',
                   'function_name': 'e224g_read_cal_codes',
                   'param_names': 'lane',
                   'params': lane}
        response = self.talk(request)
        return response

    def get_adc_samples(self, lane: int) -> dict:
        """Retreive adaptation samples
        """

        print(f'Retrieving adaptation samples on lane {lane}')
        mem_in = self.eval("e224g_sram_read()")
        request = {'operation': 'function',
                   'function_name': 'e224g_sram_convert',
                   'param_names': ['mem_in', 'capture_string'],
                   'params': [mem_in, 'adc']}
        response = self.talk(request)

        now = datetime.now()  # current date and time
        datestr = now.strftime('%m%d%Y_%H%M%S')

        with open('adc_sample_capture_' + datestr + '.csv', 'w', newline='',
                  encoding='UTF-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(response)

        return response

    def meas_snr(self, lane: int) -> float:
        """Measure SNR
        """

        return self.eval(f"e224g_get_fom({lane})")['SNR']

    def meas_ber(self, lane: int, num_reads: int = 20, test: Literal['windowed', 'confidence'] = 'windowed',
                 cl: int = 95, target: float = 0, block: str = 'tc') -> dict:
        """Measure the bit error rate of the current setup

        :param cl: Confidence level, only applicable if test == 'confidence', defaults to 95
        :param target: Target BER, only applicable if test == 'confidence', defaults to 1e-8
        """

        print(f'Measuring BER on lane {lane}')
        if test == 'windowed':
            out = self.eval_return_all(f"ber_res = e224g_find_ber({lane}, {num_reads}, {target}, '{test}', '{block}');")
            ber_results = {'errors': out['ber_res']['total_errors'],
                           'time': out['ber_res']['full_capture_time'],
                           'nbits': out['ber_res']['total_bits'],
                           'ber': out['ber_res']['ber_avg']}
        elif test == 'confidence':
            out = self.eval_return_all(f"""
            num_reads = 50;
            ber_res = e224g_find_ber({lane}, num_reads, {target}, '{test}', {cl});
            """)
            ber_results = {'ber_avg': out['ber_res']['ber_avg'],
                           'pause_time': out['ber_res']['pause_time'],
                           'full_capture_time': out['ber_res']['full_capture_time']}
        print(ber_results, end='\n')
        return ber_results
    def read_adapt_codes(self, lane:int) -> dict:
        """Retreive adapt codes
        """
        request = {'operation': 'function',
                   'function_name': 'e224g_read_adapt_codes',
                   'param_names': 'lane',
                   'params': lane}
        response = self.talk(request)
        return response
    def read_cal_codes(self, lane: int) -> dict:
        """Retreive cal codes
        """
        request = {'operation': 'function',
                   'function_name': 'e224g_read_cal_codes',
                   'params': [lane, 'output_mode', 'flat']}
        response = self.talk(request)
        return response
    def dump_sram(self, lane: int, sram_option: str) -> dict:
        """Dump SRAM
        :param sram_option: One of ['edfe_pack', 'edfe', 'adc', 'cdr_ffe', 'ffe_lower40.ffe', 'ffe_upper40.ffe']
        """
        return self.eval_return_all(f"""     
            e224g_sram_capture({lane},'{sram_option}');
            sram_raw = e224g_sram_read();
            sram_parsed = e224g_sram_convert(sram_raw,'{sram_option}');
            """)
