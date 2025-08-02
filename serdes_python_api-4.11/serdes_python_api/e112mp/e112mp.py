"""Contains the E112MP class
"""

import csv
from datetime import datetime
from typing import Literal

from serdes_python_api.shared.shared import Shared

REVISION = '$Revision: #3 $'
DATE = '$Date: 2024/05/09 $'


class E112MP(Shared):
    """Contains e112mp methods related to configuration and evaluation

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

        #Configure part
        self._set_part()

    def enable_fec(self, tx_lane: int, rx_lane: int, enable: bool) -> None:
        """Enables forward error correction
        """
        # Get fec_script script
        project = self.part_params[self.tpid-1]['project']
        fec_script = 'e112mp_' + project + '_enable_FEC'
        arg_in = {'standard': self.sp_args['standard_L']}

        match project:
            case "x590":
                request = {'operation': 'function',
                   'function_name': f'{fec_script}',
                   'params': [tx_lane, rx_lane, int(enable), arg_in]}
                self.talk(request)

            case "x588" | "x588TC2":
                request = {'operation': 'function',
                   'function_name': f'{fec_script}',
                   'params': [tx_lane, rx_lane, int(enable), arg_in]}
                self.talk(request)

            case "x589":
                request = {'operation': 'function',
                   'function_name': f'{fec_script}',
                   'params': [rx_lane, int(enable), arg_in]}
                self.talk(request)

            case "x585tc3"|"x586":
                self.eval(
                    f"e112mp_x585_TC3_enable_FEC({rx_lane}, {int(enable)}, '{self.sp_args['standard_L']}')")
            case "x585tc2":
                self.eval(
                    f"e112mp_x585_enable_FEC({rx_lane}, {int(enable)}, '{self.sp_args['standard_L']}')")
            case _:
                print('Function does not exist')

    def adapt(self, _) -> None:
        print('adapt is not currently supported for e112mp')

    def calibrate(self, _) -> None:
        print('calibrate is not currently supported for e112mp')

    def get_adaptation_results(self, lane: int) -> dict:
        """Retrieve adaptation results
        """

        print(f'Retrieving adaptation results on lane {lane}')
        request = {'operation': 'function',
                   'function_name': 'e112mp_read_adapt_codes',
                   'param_names': 'lane',
                   'params': lane}
        response = self.talk(request)
        return response

    def get_calibration_results(self, lane: int) -> dict:
        """Retrieve calibration results
        """

        print(f'Retrieving calibration results on lane {lane}')
        request = {'operation': 'function',
                   'function_name': 'e112mp_read_cal_codes',
                   'param_names': 'lane',
                   'params': lane}
        response = self.talk(request)
        return response

    def get_adc_samples(self, lane: int) -> dict:
        """Retreive adaptation samples
        """

        print(f'Retrieving adaptation samples on lane {lane}')
        mem_in = self.eval("e112mp_sram_read()")
        request = {'operation': 'function',
                   'function_name': 'e112mp_sram_convert',
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

    def meas_die_temp(self, lane: int) -> dict:
        """Measure the die temperature
        """

        return self.eval(f'e112mp_ip_meas_die_temp({lane});')

    def meas_snr(self, lane: int) -> float:
        """Measure SNR
        """

        return self.eval(f"e112mp_ip_meas_fom({lane})")['SNR']

    def meas_ber(self, lane: int, test: Literal['windowed', 'confidence'] = 'windowed',
                 cl: int = 95, target: float = 1e-8) -> dict:
        """Measure the bit error rate of the current setup

        :param cl: Confidence level, only applicable if test == 'confidence', defaults to 95
        :param target: Target BER, only applicable if test == 'confidence', defaults to 1e-8
        """

        print(f'Measuring BER on lane {lane}')
        if test == 'windowed':
            out = self.eval_return_all(f"""
            num_reads = 50;

            ber_target = 1e-8;
            ber_res = e112mp_find_ber({lane}, num_reads, ber_target, '{test}');

            if min(ber_res.errors) < 100
                ber_target = 1e-10;
                ber_res = e112mp_find_ber({lane}, num_reads, ber_target, '{test}');
            elseif max(ber_res.errors) > 64000
                ber_target = 1e-6;
                ber_res = e112mp_find_ber({lane}, num_reads, ber_target, '{test}');
            end
            """)
            ber_results = {'errors': sum(out['ber_res']['errors']),
                           'time': out['ber_res']['full_capture_time'],
                           'nbits': sum(out['ber_res']['nbits']),
                           'ber': out['ber_res']['ber_avg']}
        elif test == 'confidence':
            out = self.eval_return_all(f"""
            num_reads = 50;
            ber_res = e112mp_find_ber({lane}, num_reads, {target}, '{test}', {cl});
            """)
            ber_results = {'ber_avg': out['ber_res']['ber_avg'],
                           'pause_time': out['ber_res']['pause_time'],
                           'full_capture_time': out['ber_res']['full_capture_time']}
        print(ber_results, end='\n\n')
        return ber_results
