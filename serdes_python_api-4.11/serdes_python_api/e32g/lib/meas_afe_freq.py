"""Contains functions for measuring AFE frequency
"""

# pylint: disable=C0301, C0209, W0621, C0103

import math
import os
import time
from datetime import datetime
from typing import Any, Literal

from serdes_python_api.e32g.e32g import E32G

REVISION = '$Revision: #2 $'
DATE = '$Date: 2024/01/04 $'


def sine_generator_output_state(evb: E32G, state: Literal['ON', 'OFF']) -> None:
    """toggle sine_generator on and off
    """

    mat_code = f"""
    sgen = sine_generator;
    sgen.sine_generator_output_state('{state}');
    """
    evb.eval(mat_code)


def config_sine_generator(
        evb: E32G, action_code: Literal['X', 'Y'], var: Literal['amp', 'freq'], val: int,
        unit: Literal['V', 'mV', 'uV', 'Hz', 'kHz', 'MHz', 'GHz']) -> int:
    """Configure sine generator

    :param action_code: X: Set, Y: Get
    """

    mat_code = f"""
    sgen = sine_generator;
    val = sgen.config_sine_generator('{action_code}','{var}',{val}, '{unit}');
    """
    return evb.eval_return_all(mat_code)['val']


def afe_meas_config(evb: E32G, **kwargs) -> Any:
    """Run e32g_meas_afe_config
    """

    mat_code = f"""
    meas_params.standard = ['{kwargs['standard']}'];
    meas_params.refclk_freq = {kwargs['refclk_freq']}
    meas_params.vph = {kwargs['vph']};
    meas_params.tmp_sk_t = {kwargs['tmp_sk_t']}
    meas_params.tmp_sh_t = {kwargs['tmp_sh_t']}
    res = e32g_meas_afe_config({kwargs['lane']}, meas_params);
    """
    return evb.eval_return_all(mat_code)


def save_plots(evb: E32G, test_point, noise_signal, dac_range, afe_eq, fol_name, freq) -> None:
    """Run e32g_meas_afe_cfg_save_plot
    """

    if freq == -1:
        mat_code = f"""
        afe_eq.att = {afe_eq['att']};
        afe_eq.vga = {afe_eq['vga']};
        afe_eq.boost = {afe_eq['boost']};
        afe_eq.pole = {afe_eq['pole']};
        afe_eq.rate = {afe_eq['rate']};
        afe_eq.config = {afe_eq['config']};
        fol_name = '{fol_name}';
        e32g_meas_afe_cfg_save_plot('{test_point}', '{noise_signal}', {dac_range}, afe_eq, fol_name);
        """
    else:
        mat_code = f"""
        afe_eq.att = {afe_eq['att']};
        afe_eq.vga = {afe_eq['vga']};
        afe_eq.boost = {afe_eq['boost']};
        afe_eq.pole = {afe_eq['pole']};
        afe_eq.rate = {afe_eq['rate']};
        afe_eq.config = {afe_eq['config']};
        fol_name = '{fol_name}' ;
        e32g_meas_afe_cfg_save_plot('{test_point}', '{noise_signal}', {dac_range}, afe_eq, fol_name, {freq});
        """
    evb.eval(mat_code)


def afe_freq_meas(evb: E32G, **config_info):
    """Measure afe frequency
    """

    debug = 1
    plot_dir = config_info['plot_dir']
    lane = config_info['lane']
    meas_ctle_tf = config_info['meas_ctle_tf']
    init_amp = config_info['init_amp']
    ret = afe_meas_config(evb, **config_info)
    res = ret['res']
    att_out_slc_ranges = res['att_out_slc_ranges']
    ctle_out_slc_ranges = res['ctle_out_slc_ranges']
    ipid = evb.eval("get_pid('ip')")
    max_dac_code = 511

    # freq sweep params
    mat_code = """
    freq_b1             = logspace(7,8,5);          % 05pt, log: 10MHz - 100MHz
    freq_b2             = logspace(8,9,15);         % 15pt, log: 100MHz- 1GHz
    freq_b3             = logspace(9,10,20);        % 20pt, log: 1GHz  - 10GHz
    freq_b4             = linspace(10e9,20e9,20);   % 20pt, lin: 10GHz-20GHz
    FREQ                = unique([freq_b1 freq_b2 freq_b3 freq_b4]/1e6);
    """
    mat_ret = evb.eval_return_all(mat_code)
    freq = mat_ret["FREQ"]

    # afe tf
    if meas_ctle_tf == 1:
        vga = config_info['vga']
        boost = config_info['boost']
        afe_cfg = config_info['afe_cfg']
        afe_rate = config_info['afe_rate']
        afe_pole = config_info['afe_pole']
        if vga is None:
            vga = [evb.agr(ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_0.EQ_VGA_GAIN')]
        if boost is None:
            boost = [evb.agr(ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_0.EQ_CTLE_BOOST')]
        if afe_cfg is None:
            afe_cfg = [evb.agr(ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_10.RX_AFE_CONFIG')]
        if afe_rate is None:
            afe_rate = [evb.agr(ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_1.EQ_AFE_RATE')]
        if afe_pole is None:
            afe_pole = [evb.agr(ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_1.EQ_CTLE_POLE')]

        evb.asr(ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_0.EQ_ATT_LVL', 0)
        for xvga in vga:
            for xboost in boost:
                for xafe_cfg in afe_cfg:
                    for xafe_rate in afe_rate:
                        for xafe_pole in afe_pole:
                            total_time_elapsed = 0
                            # ----------------------------------------------
                            # Configure CTLE Eq settings
                            # ----------------------------------------------
                            evb.asr(ipid, 'LANE' + str(lane) +
                                    '.ASIC_RX_OVRD_EQ_IN_0.EQ_VGA_GAIN', xvga)
                            evb.asr(ipid, 'LANE' + str(lane) +
                                    '.ASIC_RX_OVRD_EQ_IN_0.EQ_CTLE_BOOST', xboost)
                            evb.asr(ipid, 'LANE' + str(lane) +
                                    '.ASIC_RX_OVRD_EQ_IN_10.RX_AFE_CONFIG', xafe_cfg)
                            evb.asr(ipid, 'LANE' + str(lane) +
                                    '.ASIC_RX_OVRD_EQ_IN_1.EQ_AFE_RATE', xafe_rate)
                            evb.asr(ipid, 'LANE' + str(lane) +
                                    '.ASIC_RX_OVRD_EQ_IN_1.EQ_CTLE_POLE', xafe_pole)

                            # ----------------------------------------------
                            # Update CTLE Eq Settings
                            # ----------------------------------------------
                            ack_status = evb.eval(f"""e32g_pma_req_ack_handshake({lane},'RX')""")
                            if ack_status['rx_ack_1'] == 0:
                                print('RX EQ UPDATE:  RX_ACK ' + str(lane) + '0->1 timed out.\n')
                            if ack_status['rx_ack_0'] == 1:
                                print('RX EQ UPDATE:  RX_ACK ' + str(lane) + '1->0 timed out.\n')

                            # ----------------------------------------------
                            # Run AFE calibration
                            # ----------------------------------------------
                            sine_generator_output_state(evb, 'OFF')
                            mat_code = f"""
                            meas_params.run_cal = 1';
                            meas_params.run_att_cal = 1;
                            e32g_mm_rx_afe_cal_fast({ipid}, {lane}, meas_params);
                            """
                            evb.eval(mat_code)

                            sine_generator_output_state(evb, 'ON')

                            # ----------------------------------------------
                            # Get current AFE Eq settings, calibration
                            # codes, vreg_clk
                            # ----------------------------------------------
                            afe_eq = {}
                            afe_eq['att'] = evb.agr(
                                ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_0.EQ_ATT_LVL')
                            afe_eq['vga'] = evb.agr(
                                ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_0.EQ_VGA_GAIN')
                            afe_eq['vga'] = afe_eq['vga'] >> 1
                            afe_eq['boost'] = evb.agr(
                                ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_0.EQ_CTLE_BOOST')
                            afe_eq['pole'] = evb.agr(
                                ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_1.EQ_CTLE_POLE')
                            afe_eq['rate'] = evb.agr(
                                ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_1.EQ_AFE_RATE')
                            afe_eq['config'] = evb.agr(
                                ipid, 'LANE' + str(lane) + '.ASIC_RX_OVRD_EQ_IN_10.RX_AFE_CONFIG')
                            cal_codes = evb.eval(f"""get_cal_codes({lane})""")
                            rx_att_cal_code = cal_codes['rx_att_cal_code']
                            rx_buf_cal_code = cal_codes['rx_buf_cal_code']
                            rx_ctle_cal_code = cal_codes['rx_ctle_cal_code']
                            rx_vga1_cal_code = cal_codes['rx_vga1_cal_code']
                            rx_afe_trim_code = cal_codes['rx_afe_trim_code']
                            vreg_clk_0 = evb.agr(ipid, 'LANE' + str(lane) +
                                                 '.RX_SLC_CTRL.VREG_BOOST_REG_6')
                            vreg_clk_1 = evb.agr(ipid, 'LANE' + str(lane) +
                                                 '.RX_SLC_CTRL.VREG_BOOST_REG_7')
                            vreg_clk = vreg_clk_1 << 1 + vreg_clk_0

                            # ----------------------------------------------
                            # Create a folder to store signal CDF's plots
                            # ----------------------------------------------
                            if debug == 1:
                                fol_name1 = 'ATT' + str(afe_eq['att']) + '_BST' + str(afe_eq['boost']) + '_VGA' + str(
                                    afe_eq['vga']) + '_R' + str(afe_eq['rate']) + '_CFG' + str(afe_eq['config'])
                                fol_name = fol_name1 + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
                                # plot_dir = "C:/Users/us03lab/Perforce/akhandel_us03alab-b6/SERDES/temp/afe_feq_test"
                                fol_name = os.path.join(plot_dir, fol_name)
                                print(fol_name)
                                os.mkdir(fol_name)

                            # ----------------------------------------------
                            # Measure small signal noise at CTLE input
                            # ----------------------------------------------
                            sigma = 0
                            meas_params = evb.eval(
                                f"""e32g_meas_afe_amp_and_noise({ipid}, {lane}, 'ATT_OUT', 'NOISE', {sigma}, {debug})""")
                            att_out_noise_sigma = meas_params['sigma']
                            att_out_noise_error = meas_params['fmin']

                            if debug:
                                save_plots(evb, 'ATT_OUT', 'NOISE', 1, afe_eq, fol_name, -1)

                            # ----------------------------------------------
                            # Measure small signal noise at CTLE output
                            # ----------------------------------------------
                            meas_params = evb.eval(
                                f"""e32g_meas_afe_amp_and_noise({ipid}, {lane}, 'CTLE_OUT', 'NOISE', {sigma}, {debug})""")
                            ctle_out_noise_sigma = meas_params['sigma']
                            ctle_out_noise_error = meas_params['fmin']

                            if debug:
                                save_plots(evb, 'CTLE_OUT', 'NOISE', 1, afe_eq, fol_name, -1)

                            # ----------------------------------------------
                            # Set initial amplitude
                            # ----------------------------------------------
                            config_sine_generator(evb, 10, 'amp', init_amp, 'mV')

                            # ----------------------------------------------
                            # Sweep Frequency
                            # ----------------------------------------------
                            for f in freq:
                                t0 = time.time()
                                # ----------------------------------------------------------
                                # Set SINE PG frequency
                                # ----------------------------------------------------------
                                freq_set = config_sine_generator(evb, 11, 'freq', f, 'MHz')

                                # ----------------------------------------------------------
                                # SHIFT VCO freq if current sine freq is the same VCO freq
                                # ----------------------------------------------------------
                                if freq_set == 8e9:
                                    evb.asr(ipid, 'LANE' + str(lane) + '.RX_DPLL_FREQ.VAL',
                                            0)        # RX VCO FREQ = ~8GHz - 50kppm
                                else:
                                    evb.asr(ipid, 'LANE' + str(lane) + '.RX_DPLL_FREQ.VAL',
                                            8192)     # RX VCO FREQ = ~8GHz

                                # ----------------------------------------------------------
                                # Find input amp/scope_range
                                # ----------------------------------------------------------
                                mat_code = f"""
                                slc_ranges.att_out = {att_out_slc_ranges};
                                slc_ranges.ctle_out = {ctle_out_slc_ranges};
                                sigmas.att_out      = {att_out_noise_sigma};
                                sigmas.ctle_out     = {ctle_out_noise_sigma};
                                [amp_set, att_out_slc_range, ctle_out_slc_range, cal_info] = e32g_meas_afe_find_amp_and_slc_range({ipid},{lane},slc_ranges,sigmas);
                                """
                                ret = evb.eval_return_all(mat_code)
                                amp_set = ret['amp_set']
                                att_out_slc_range = ret['att_out_slc_range']
                                ctle_out_slc_range = ret['ctle_out_slc_range']
                                cal_info = ret['cal_info']
                                # ----------------------------------------------------------
                                # Measure signal amplitude at ATT_OUT
                                # ----------------------------------------------------------

                                att_out_meas_params = evb.eval(
                                    f"""e32g_meas_afe_amp_and_noise({ipid}, {lane}, 'ATT_OUT', 'SIGNAL', {att_out_noise_sigma}, {debug})""")
                                att_out_meas_params['amp_mV'] = att_out_meas_params['amp'] * \
                                    att_out_slc_ranges[att_out_slc_range - 1]/max_dac_code

                                if debug:
                                    save_plots(evb, 'ATT_OUT', 'SIGNAL', att_out_slc_range, afe_eq,
                                               fol_name, freq_set)

                                # ----------------------------------------------------------
                                # Measure signal amplitude at CTLE_OUT
                                # ----------------------------------------------------------
                                ctle_out_meas_params = evb.eval(
                                    f"""e32g_meas_afe_amp_and_noise({ipid}, {lane}, 'CTLE_OUT', 'SIGNAL', {ctle_out_noise_sigma}, {debug})""")
                                ctle_out_meas_params['amp_mV'] = ctle_out_meas_params['amp'] * \
                                    ctle_out_slc_ranges[ctle_out_slc_range]/max_dac_code

                                if debug:
                                    save_plots(evb, 'CTLE_OUT', 'SIGNAL', ctle_out_slc_range, afe_eq,
                                               fol_name, freq_set)
                                # ----------------------------------------------------------
                                # Measure CTLE gain
                                # ----------------------------------------------------------
                                ctle_gain = 20 * \
                                    math.log(
                                        (ctle_out_meas_params['amp_mV']/att_out_meas_params['amp_mV']), 10)
                                t1 = time.time()
                                time_elapsed = t1 - t0

                                # ----------------------------------------------------------
                                # Store data to file
                                # ----------------------------------------------------------
                                with open(config_info['f_name'], 'a', encoding='utf-8') as fid:
                                    # Equipment/setup information
                                    print('%d\t%.3f\t%.2f\t' %
                                          (freq_set, amp_set, time_elapsed), file=fid)
                                    # AFE Eq settings info
                                    print('%i\t%i\t%i\t%i\t%i\t%i\t%i\t' % (
                                        afe_eq['att'], afe_eq['vga'], afe_eq['boost'], afe_eq['pole'], afe_eq['rate'], afe_eq['config'], vreg_clk), file=fid)
                                    # AFE CAL Codes
                                    print('%i\t%i\t%i\t%i\t%i\t' % (rx_afe_trim_code, rx_att_cal_code,
                                          rx_buf_cal_code, rx_ctle_cal_code, rx_vga1_cal_code), file=fid)
                                    # ATT/CTLE_OUT amplitude info
                                    print('%.2f\t%.2f\t%.2f\t%s\t' % (
                                        att_out_meas_params['amp_mV'], ctle_out_meas_params['amp_mV'], ctle_gain, ' '), file=fid)

                                    # DAC ranges
                                    print('%i\t%.1f\t%.1f\t%.1f\t' % (
                                        att_out_slc_range, att_out_slc_ranges[0], att_out_slc_ranges[1], att_out_slc_ranges[2]), file=fid)
                                    print('%i\t%.1f\t%.1f\t%.1f\t' % (
                                        ctle_out_slc_range, ctle_out_slc_ranges[0], ctle_out_slc_ranges[1], ctle_out_slc_ranges[2]), file=fid)

                                    # AMP CAL info
                                    print('%i\t%i\t' %
                                          (cal_info['cal_case'], cal_info['high_sig_amp']), file=fid)

                                    # ATT/CTLE_OUT Noise
                                    print('%.2f\t%.2f\t%.2f\t%.2f\t' % (
                                        att_out_noise_sigma, att_out_noise_error, ctle_out_noise_sigma, ctle_out_noise_error), file=fid)

                                    # ATT_OUT Debug info
                                    print('%.2f\t%.2f\t%.2f\t%.2f\t' % (
                                        att_out_meas_params['amp'], att_out_meas_params['sigma'], att_out_meas_params['sig_offset'], att_out_meas_params['fit_err']), file=fid)
                                    print('%.2f\t%.2f\t' % (
                                        att_out_meas_params['amp1'], att_out_meas_params['amp2']), file=fid)
                                    print('%.2f\t%.2f\t' % (
                                        att_out_meas_params['sigma1'], att_out_meas_params['sigma2']), file=fid)
                                    print('%.2f\t%.2f\t' % (
                                        att_out_meas_params['sig_offset1'], att_out_meas_params['sig_offset2']), file=fid)
                                    print('%.2f\t%.2f\t%.2f\t' % (
                                        att_out_meas_params['fmin1'], att_out_meas_params['fmin2'], att_out_meas_params['fmin3']), file=fid)

                                    # CTLE_OUT Debug info
                                    print('%.2f\t%.2f\t%.2f\t%.2f\t' % (
                                        ctle_out_meas_params['amp'], ctle_out_meas_params['sigma'], ctle_out_meas_params['sig_offset'], ctle_out_meas_params['fit_err']), file=fid)
                                    print('%.2f\t%.2f\t' % (
                                        ctle_out_meas_params['amp1'], ctle_out_meas_params['amp2']), file=fid)
                                    print('%.2f\t%.2f\t' % (
                                        ctle_out_meas_params['sigma1'], ctle_out_meas_params['sigma2']), file=fid)
                                    print('%.2f\t%.2f\t' % (
                                        ctle_out_meas_params['sig_offset1'], ctle_out_meas_params['sig_offset2']), file=fid)
                                    print('%.2f\t%.2f\t%.2f\t' % (
                                        ctle_out_meas_params['fmin1'], ctle_out_meas_params['fmin2'], ctle_out_meas_params['fmin3']), file=fid)

                                    print('\n', file=fid)
                                    fid.close()

                                print('CAL_CASE=%i, HIGH_SIG=%i --- ' %
                                      (cal_info['cal_case'], cal_info['high_sig_amp']))
                                print('Freq=%0.4e MHz, ATT=%i, VGA=%02i, BOOST=%02i, GAIN=%04.2fdB, Time=%2.1fs\n' % (
                                    freq_set/1e6, afe_eq['att'], afe_eq['vga'], afe_eq['boost'], ctle_gain, time_elapsed))
                                total_time_elapsed = total_time_elapsed + time_elapsed
                            print('total time elapsed: %f' % (total_time_elapsed))


if __name__ == '__main__':
    # general configuration parameters
    standard = 'PCIE5_CC'
    lane = 0
    pattern = 'prbs31'
    refclk_freq = -1  # use default freq

    # sinking time
    tmp_sk_t = 1
    tmp_sh_t = 1

    # afe parameters
    vga = [15, 16]  # None #0-31
    boost = None  # 0-31
    afe_cfg = None  # 0-4096
    afe_rate = None  # 0-7
    afe_pole = None  # 0 - 3

    # voltages
    vph = 1.8
    init_amp = 45

    # log file
    f_name = 'FS202_ctle_tf_vt_sweep.txt'

    # measure ctle tf
    meas_ctle_tf = 1

    config_info = {}

    config_info['standard'] = standard
    config_info['lane'] = lane
    config_info['pattern'] = pattern
    config_info['refclk_freq'] = refclk_freq

    config_info['vph'] = vph
    config_info['tmp_sk_t'] = tmp_sk_t
    config_info['tmp_sh_t'] = tmp_sh_t

    config_info['vga'] = vga
    config_info['boost'] = boost
    config_info['afe_cfg'] = afe_cfg
    config_info['afe_rate'] = afe_rate
    config_info['afe_pole'] = afe_pole
    config_info['f_name'] = f_name
    config_info['meas_ctle_tf'] = meas_ctle_tf
    config_info['init_amp'] = init_amp
    config_info['plot_dir'] = 'C:/Users/Documents'
    # change above path as per requirement

    if config_info['plot_dir'] is None:
        print('Please provide full path of output directory')
        exit()

    evb = E32G()
    afe_freq_meas(evb, **config_info)
