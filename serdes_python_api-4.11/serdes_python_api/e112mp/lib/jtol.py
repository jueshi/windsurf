"""Module contains the implementation of the Jtol class.
"""

import os
import time
import pandas as pd
from serdes_python_api.e112mp.evb import Evb
from serdes_python_api.e112mp.lib.bert import Bert

REVISION = '$Revision: #3 $'
DATE = '$Date: 2022/12/22 $'


class Jtol():
    """Responsible for executing the JTOL search.
    """

    def __init__(self, protocol: str) -> None:
        self.protocol = protocol
        self.evb = Evb()

    def run(self, bert: Bert, lane: int, p: pd.Series, plot: bool) -> dict:
        """Executes 3 search loops, then stores the final amplitude. Returns after
        iterating over all frequencies.

        :param p: container with all the required information to run jtol
        :return: relevant results and part data
        """

        rv = self._configure_part(lane, p, plot)
        rv['tx_eq'] = bert.get_tx_eq()

        for i, (kind, freq, amp) in enumerate(zip(p.types, p.freqs, p.amps)):
            # reset existing frequency
            bert.set_frequency('PER', 0)
            bert.set_frequency('SIN', 0)

            # reset existing amplitude
            bert.set_amplitude('PER', 0)
            bert.set_amplitude('SIN', 0)

            # set the frequency and amplitude
            bert.set_frequency(kind, freq)
            bert.set_amplitude(kind, amp)

            # cdr
            self.evb.eval(
                f"e112mp_ip_meas_cdr({lane}, struct('plot_en', 0))")

            wait_c = (1 / p.baud_rate) / p.ber_coarse
            step_c = p.step_ratio_coarse * amp
            pass_c = self._sweep(bert, lane, amp, freq,
                                 wait_c, step_c, kind, plot, p.coding, p.ber_coarse)

            wait_m = (1 / p.baud_rate) / p.ber_medium
            step_m = p.step_ratio_medium * amp
            pass_m = self._sweep(bert, lane, max(0, pass_c - step_m),
                                 freq, wait_m, step_m, kind, plot, p.coding, p.ber_medium)

            wait_f = (1 / p.baud_rate) / p.ber_fine
            step_f = p.step_ratio_fine * p.amps[i]
            pass_f = self._sweep(bert, lane, max(0, pass_m - step_f),
                                 freq, wait_f, step_f, kind, plot, p.coding, p.ber_fine)

            rv[f'#{i}']['passing pj'] = pass_f

            # update the plot with the passing pj, and force a render
            if plot:
                self.evb.eval(f"live_plot({freq}, {pass_f}, 'final'); drawnow()")
        return rv

    def _sweep(self, b, lane, start_amp, freq, wait, step, kind, plot, coding, sweep_ber) -> float:
        """Iterate over and return the highest error-free amplitude.

        :param s: Session object
        :param b: Bert object
        :param wait: how long to continuously check for errors
        :param step: how much to adjust amplitude each iteration
        :param kind: PER or SIN
        :param plot: Enable plot
        :param coding: 'PAM4', or 'NRZ'
        :param sweep_ber: ber level for the sweep
        :return: passing amplitude
        """

        curr_amp = start_amp
        max_amp = b.get_max_amplitude(kind)
        while True:
            b.set_amplitude(kind, curr_amp)

            # cdr
            self.evb.eval(
                f"e112mp_ip_meas_cdr({lane}, struct('plot_en', 0))")

            if coding == 'NRZ':
                error_count = self._get_error_count(lane, wait)
                success = error_count == 0
                if 'direction' not in locals():
                    direction = self._get_direction_nrz(error_count)
            elif coding == 'PAM4':
                ber = self.evb.meas_ber(rx_lanes=[1])[1]['ber']
                success = ber < sweep_ber
                if 'direction' not in locals():
                    direction = self._get_direction_pam4(ber, sweep_ber)

            # update the direction of the arrow in the plot, and the amplitude
            if plot:
                self.evb.eval(f"live_plot({freq}, {curr_amp}, '{direction}');")

            # terminal output
            print(f"| Frequency: {freq:.0e}, PJ type: '{kind}', "
                  f"PJ amp UI: {curr_amp:.2f}")
            if coding == 'NRZ':
                print(f"| Direction: {direction}, Error count: {error_count}", sep='')
            elif coding == 'PAM4':
                print(f"| Direction: {direction}, BER: {ber}", sep='')
            print('|____________________________________________________')

            if direction == 'up':  # sweep upwards
                if success:
                    if curr_amp == max_amp:  # reached max_amp, cannot continue (pass)
                        return curr_amp
                    else:
                        curr_amp += step
                        if curr_amp > max_amp:
                            curr_amp = max_amp
                else:  # return the previous error free amp (fail)
                    return curr_amp - step
            elif direction == 'down':  # sweep downwards
                if success:
                    return curr_amp  # return error free amp (pass)
                else:
                    if curr_amp == 0:  # reached min amp, cannot continue (fail)
                        return curr_amp
                    else:
                        curr_amp -= step
                        if curr_amp < 0:
                            curr_amp = 0

    def _get_direction_pam4(self, ber: float, sweep_ber) -> str:
        """If the ber is less than the sweep ber, go up, else, go down.

        :return: direction
        """

        if ber < sweep_ber:
            return 'up'
        else:
            return 'down'

    def _get_direction_nrz(self, error_count: int) -> str:
        """If there are no errors, sweep up, else, sweep down.

        :return: direction
        """

        if error_count == 0:
            return 'up'
        else:
            return 'down'

    def _configure_part(self, lane: int, p: pd.Series, plot) -> dict:
        """Bring up the part, and store configuration data in a return value that
        is going to be used later to store the passing pj.

        :return: relevant part data
        """

        # dictionary comprehension to create nested dictionary
        rv = {f'#{i}': {'sj_frequency': f'{freq:.0e}', 'pj_type': type}
              for i, (freq, type) in enumerate(zip(p.freqs, p.types))}

        # initialize the params on the matlab side, then create the plot
        if plot:
            self.evb.eval(f"tol_params('{self.protocol}'); live_plot('init');")

        self.evb.initialize(**self.evb.sp_args)

        # save startup codes and error
        rv['startup_cal_codes'] = self.evb.eval(
            f'e112mp_read_cal_codes({lane})')
        rv['startup_adapt_codes'] = self.evb.eval(
            f'e112mp_read_adapt_codes({lane})')

        return rv

    def get_params(self, protocol: str) -> pd.Series:
        """Returns the params respective to standard.

        :param protocol: pcie4, pcie5, etc
        :return: required jtol parameters
        """

        def f1(x):
            return list(map(float, x.split()))

        df = pd.read_csv(os.getcwd() + '\\jtol_params.tsv', sep='\t',
                         converters={'freqs': f1,
                                     'amps': f1, 'types': str.split},
                         index_col=False)

        return df[df['protocol'] == protocol].iloc[0, :]

    def _get_error_count(self, lane, wait) -> int:
        """Syncs the pm, then checks for errors.

        :return: error count
        """
        error_count = 0
        sync = self.evb.eval(f"ip_cfg_sync_pm({lane})")
        t = time.time()
        while ((time.time() - t) < wait) and error_count == 0:
            error_count = self.evb.eval(f"error_check({lane}, struct('sync_timer', "
                                        f"uint64({sync['sync_timer']}), 'ber_sync_valid', "
                                        f"{sync['ber_sync_valid']}, 'inverted_pattern', 0));")
        return error_count

    def configure_bert(self, b: Bert, p: pd.Series, std: str) -> None:
        """Set the bert to a reset state before running JTOL.

        :param s: socket connection between python (client) and matlab (server)
        :param b: Bert object used to interface with the physical bert
        :param std: standard - pcie5_cc, pcie5_sris, etc
        """

        b.set_baudhz(p.baud_rate)
        b.set_amplitude('PER', 0)
        b.set_amplitude('SIN', 0)
        b.toggle_data('on')
        b.toggle_data('off')
        b.set_ssc(std)
        b.toggle_data('on')
