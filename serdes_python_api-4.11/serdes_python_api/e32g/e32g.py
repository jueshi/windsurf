"""Contains the E32G class
"""

import csv
import math
from datetime import datetime
from typing import Literal

from serdes_python_api.shared.shared import Shared

REVISION = '$Revision: #6 $'
DATE = '$Date: 2024/08/14 $'


class E32G(Shared):
    """Contains e32g methods related to configuration and evaluation

    :param Shared: Parent class containing product generic methods
    """

    def initialize(self, clock_args: dict | None = None) -> None:
        """Comprehensive configuration procedure

        :param clock_args: arguments for set_clock, defaults to None
        """

        print('Initializing EVB')

        # clear adapt codes
        self.eval("clear_adapt_codes(get_pid('ip'))")

        # initialize clock chip
        if clock_args is not None:
            self.clock_args = clock_args
        self._set_clock()
        self._set_part()

        for lane, enable in enumerate(self.sp_args['rx_en_L']):
            if enable:
                self.adapt(lane)

    def adapt(self, lane: int) -> None:
        """Runs adaptation if the standard supports it

        :param lane: Data path
        """

        if self.protocol_defaults['rx_dfe_adaptable'] or self.protocol_defaults['rx_afe_adaptable']:
            args = {}
            if not self.protocol_defaults['rx_dfe_adaptable']:
                args['rx_adapt_afe_ovrd'] = 1

            self.func('rx_startup_adaptation', self.ipid, lane, args)
            self.sync_pm(lane)

    def meas_cdr(self, lane: int) -> dict:
        """Measure clock data recovery (CDR)

        :param lane: Data path
        :return: CDR properties
        """

        return self.func('e32g_ip_meas_cdr', lane)[0]

    def meas_dac_ranges(self, lane: int) -> dict:
        """Measure DAC ranges

        :param lane: Data path
        :return: DAC ranges
        """

        return self.func('e32g_meas_dac_ranges', self.ipid, lane)

    def get_polarity(self, tx_lane: int, rx_lane: int) -> Literal[-1, 0, 1]:
        """Return the polarity of the connection from tx_lane and rx_lane

        :param tx_lane: Transmitter data path
        :param rx_lane: Receiver data path
        :return: 0 if normal, 1 if inverted, -1 if could not find
        """

        return self.func('get_polarity', [self.ipid] * 2, [tx_lane, rx_lane],
                         {'numErrors': 1, 'errorThresh': 800})

    def meas_ber(self, lane: int, ber_target: float = 1e-12, conf_level: int = 95) -> float:
        """Calculate the bit error rate (BER)

        :param lane: Data path
        :param ber_target: BER measurement target (floor), defaults to 1e-12
        :param conf_level: Confidence level (%)
        :return: BER
        """

        print(f'BER calculation in progress for lane {lane}')
        ber = self.func('find_BER', self.ipid, lane,
                        {'BER': ber_target, 'conf_lvl': conf_level})['actual_BER']
        print(f'Actual BER is {ber}', end='\n\n')
        return ber

    def get_calibration_results(self, lane: int) -> dict:
        """Retrieve calibration results
        """

        print(f'Retrieving calibration results on lane {lane}')
        ret = self.func('e32g_get_rx_cal_codes', lane)
        return ret
    
    def get_adaptation_results(self, lane: int) -> None:
        """Print adaptation results to file

        :param lane: Data path
        """
        output = {}
        print(f'Retrieving adaptation results for lane {lane}')
        ret = self.func('e32g_get_adapt_codes', lane)
        # add standard and pattern
        standard_pattern_info = {
            'standard': self.sp_args['standard_L'], 'pattern': self.sp_args['pattern_L']}
        output.update(standard_pattern_info)
        output.update(ret)
        return output

    def run_cca(self, lane: int) -> None:
        """Run continuous calibration and adaptation

        :param lane: Data path
        """

        print(f'CCA loop running for lane {lane}')
        if not self.protocol_defaults['rx_dfe_adaptable']:
            self.func('e32g_fw_mm', self.ipid, lane, {
                      'run_cal': 1, 'run_adapt': 1, 'rx_dfe': {'rx_adapt_afe_ovrd': 1}})
        else:
            self.func('e32g_test_mm_fast', self.ipid, lane)
        print('CCA loop done')

    def set_pattern(self, lane: int, pattern: str) -> None:
        """Set the pattern

        :param lane: Data path
        :param pattern: Pattern for the Patter Generator/Matcher
        """

        self.func('e32g_ip_cfg_set_pattern', lane, pattern, 'TXRX')
        self.func('relock_cdr', lane)
        self.adapt(lane)
        self.sync_pm(lane)
        errors = self.count_error(lane)
        print(f'Error count = {errors}')
        print(f'Pattern: {pattern} configured \n')

    def set_tx_eq(self, tx_lane: int, rx_lane: int, pre: int, post: int, main: int) -> None:
        """Set the tx equalization

        :param tx_lane: Transmitter data path
        :param rx_lane: Receiver data path
        :param pre: Cursor #-1
        :param post: Cursor #1
        :param main: Cursor #0
        """

        self.func('e32g_setup_txeq', self.ipid, tx_lane, pre, post, main)
        self.func('relock_cdr', rx_lane)
        self.adapt(rx_lane)
        self.sync_pm(rx_lane)
        errors = self.count_error(rx_lane)
        print(f'Error count = {errors}\n')

    def tx_eq_sweep(self, tx_lane: int, rx_lane: int, ber_duration_sec: int) -> None:
        """Run a TX equalization sweep across the pre, main, and post cursors

        :param tx_lane: Transmitter data path
        :param rx_lane: Receiver data path
        :param ber_duration_sec: how long to measure BER for
        """

        output_filename = f'e32g_txeq_sweep_lane_{rx_lane}.csv'
        columns = ['Standard', 'Pattern', 'pre', 'main', 'post', 'cycle_time',
                   'cycle_error', 'cycle_BER', 'total_error', 'total_BER', 'FOM_AFE']
        if self.protocol_defaults['rx_afe_adaptable']:
            columns.extend(['FOM', 'iq_left_right'])

        with open(output_filename, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()

        print('\n\n***************************************')
        print('Running Measurement Tests')
        print('***************************************')

        # find best tx-eq coarse search
        loop_count = 0
        total_errors = 0
        total_time = 0
        total_ber = 0
        tx_eq_res = {}
        overflowed = False
        adapt_args = self.dict_to_mstruct(
            {'rx_adapt_afe_ovrd': int(not self.protocol_defaults['rx_dfe_adaptable'])})
        for pre in range(0, 63, 4):
            for post in range(0, 63, 4):
                main = 24 - (math.floor(pre/4) + math.floor(post/4))
                if main < 0:
                    continue
                mat_out = self.eval_return_all(
                    f"""
                    e32g_setup_txeq(get_pid('ip'), {tx_lane}, {pre}, {post}, {main})
                    res.pre = {pre};
                    res.main = {main};
                    res.post = {post};
                    relock_cdr({rx_lane});
                    if {int(self.protocol_defaults['rx_afe_adaptable'])}
                        adapt_res = rx_startup_adaptation(get_pid('ip'),{rx_lane}, {adapt_args});
                        if {int(not self.protocol_defaults['rx_dfe_adaptable'])}
                            res.fom = '-';
                            res.iq_left_right = '-';
                            res.fom_afe = adapt_res.lane_rx({rx_lane} + 1).fom_afe_eyeh
                        elseif isempty(adapt_res)
                            res.fom = 0;
                            res.fom_afe = 0;
                            res.iq_left_right = 0;
                            res.iq_right = 0;
                        else
                            res.fom = adapt_res.lane_rx({rx_lane} + 1).fom;
                            res.fom_afe = adapt_res.lane_rx({rx_lane} + 1).fom_afe_eyeh
                            res.iq_left_right = adapt_res.lane_rx({rx_lane} + 1).adapt_iq_left + ...
                            adapt_res.lane_rx({rx_lane} + 1).adapt_iq_right;
                        end
                    else
                        res.fom = '-';
                        res.iq_left_right = '-';
                        res.fom_afe = '-';
                    end

                    ip_cfg_sync_pm({rx_lane});
                    test_timer = tic;
                    pause({ber_duration_sec});
                    time_elapsed = toc(test_timer);
                    res.time_elapsed = time_elapsed;
                    [err, overflow, ~] = ip_cfg_count_error({rx_lane});
                    res.cycle_errors = err;
                    res.cycle_nbits = part_params(get_pid('ip')).config.lane({rx_lane}+1).settings.baudrate * 1e9 * (time_elapsed);
                    res.cycle_ber = res.cycle_errors/res.cycle_nbits;
                    if overflow
                        res.cycle_ber = NaN;
                    end
                    """
                )
                tx_eq_res[loop_count] = mat_out['res']

                print('==================================================================')
                print(f'Eq Coarse Cycle : {loop_count} \n')

                fom = tx_eq_res[loop_count]['fom']
                fom_afe = tx_eq_res[loop_count]['fom_afe']
                time_elapsed = tx_eq_res[loop_count]['time_elapsed']
                cycle_errors = tx_eq_res[loop_count]['cycle_errors']
                cycle_ber = tx_eq_res[loop_count]['cycle_ber']
                standard = self.sp_args['standard_L']
                pattern = self.sp_args['pattern_L']
                if cycle_ber is None:
                    cycle_ber = 'None'
                    overflowed = True
                total_time = total_time + time_elapsed
                total_errors = total_errors + cycle_errors
                total_ber = total_errors / \
                    (total_time *
                     self.eval(f"part_params(get_pid('ip')).config.lane({rx_lane}+1).settings.baudrate * 1e9;"))
                if overflowed:
                    total_ber = 'None'
                iq_left_right = tx_eq_res[loop_count]['iq_left_right']
                loop_count = loop_count + 1
                if self.protocol_defaults['rx_afe_adaptable'] or \
                        self.protocol_defaults['rx_dfe_adaptable']:
                    print(f'pre = {pre}, main = {main}, post = {post}, '
                          f'Cycle BER = {cycle_ber}, fom = {fom}, fom_afe = {fom_afe}')
                    # WRITE DATA to CSV
                    rep = {}
                    rep['Standard'] = standard
                    rep['Pattern'] = pattern
                    rep['cycle_time'] = time_elapsed
                    rep['pre'] = pre
                    rep['main'] = main
                    rep['post'] = post
                    rep['cycle_error'] = cycle_errors
                    rep['cycle_BER'] = cycle_ber
                    rep['FOM'] = fom
                    rep['FOM_AFE'] = fom_afe
                    rep['iq_left_right'] = iq_left_right
                    rep['total_error'] = total_errors
                    rep['total_BER'] = total_ber
                else:
                    print(f'pre = {pre}, main = {main}, post = {post}, Cycle BER = {cycle_ber} \n')
                    # # WRITE DATA to CSV
                    rep = {}
                    rep['Standard'] = standard
                    rep['Pattern'] = pattern
                    rep['pre'] = pre
                    rep['main'] = main
                    rep['post'] = post
                    rep['cycle_time'] = time_elapsed
                    rep['cycle_error'] = cycle_errors
                    rep['cycle_BER'] = cycle_ber
                    rep['total_error'] = total_errors
                    rep['total_BER'] = total_ber

                with open(output_filename, 'a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=columns)
                    writer.writerow(rep)
