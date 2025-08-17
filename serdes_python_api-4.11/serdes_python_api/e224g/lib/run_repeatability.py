"""Contains rx repeatability function using 
"""

import csv
import time
import math
from datetime import datetime

from serdes_python_api.e224g.e224g import evb
from serdes_python_api.shared.lib.csv_utils import append_dict_to_csv

REVISION = '$Revision: #1 $'
DATE = '$Date: 2024/01/04 $'


def run_repeatability(sp_args_rx: dict | None = None, rx_input_params: dict|None = None) -> None:
    #TX Parameters
    tx_part_name = rx_input_params['tx_part_name']
    tx_swing    = rx_input_params['swing']
    cm1_list    = list(map(lambda x: (-1) * x/64, rx_input_params['cm1_list']))
    cm2_list    = list(map(lambda x: x/64, rx_input_params['cm2_list']))
    c1_list     = [0]

    #RX Parameters
    rx_part_name            = rx_input_params['rx_part_name']
    file_name               = rx_input_params['file_name']
    ctle_list               = rx_input_params['ctle_list']
    mf_list                 = rx_input_params['mf_list']
    adpt_type_list          = rx_input_params['adpt_type_list']
    pra_list                = rx_input_params['pra_list']
    sp_args_rx['run_cdr_recentre_algo']  = rx_input_params['run_cdr_recentre_algo']
    sp_args_rx['cdr_disable_int_path']   = rx_input_params['dis_int_path']

    if rx_input_params['voltage_corner'] == "nom":
        lim1 = 1
        lim2 = 1
    elif rx_input_params['voltage_corner'] == "low":
        lim1 = 0.95
        lim2 = 0.9
    elif rx_input_params['voltage_corner'] == "high":
        lim1 = 1.05
        lim2 = 1.1

    #Define voltage dictionary based on corner
    v = {}
    v['vpdig'] = 0.75 * lim2
    v['vdtx']= 1.2 * lim1
    v['vprx']= 0.8 * lim1
    v['vpcm']= 0.75 * lim1
    v['avtt']= 1.5 * lim1
    v['vphrx'] = 1.5 * lim1
    v['vphcm'] = 1.2 * lim1

    '''=========================================================================
       TX Bring Up
       ========================================================================='''
    evb.select_dongle('TX')

    evb.eval(
            f"""
            evb = EVB_sm1b;
            evb.write_vdd7_regulator('VOUT_CMD',{v['vpdig']});
            evb.set_vdd4_regulator({v['vdtx']});
            evb.set_vdd2_regulator({v['vprx']});
            evb.set_avdd2_regulator({v['vpcm']});
            evb.set_vdd6_regulator({v['avtt']});
            evb.set_vdd3_regulator({v['vphrx']});
            evb.set_avdd1_regulator({v['vphcm']});
            """
        )


    '''=========================================================================
       Configure TX
       =========================================================================
    '''
    sp_in_tx = {}
    lane_list = [0, 0, 0, 0]
    rx_lane = rx_input_params['lane']
    lane_list[rx_input_params['lane']] = 1

    sp_in_tx['standard_L']  = '200GBASE-KR-531M'
    sp_in_tx['pattern_L']   = 'prbs31'
    sp_in_tx['fw_version']  = '2p0p13'
    sp_in_tx['tx_en_L']     = lane_list
    sp_in_tx['rx_en_L']     = [0, 0, 0, 0]
    sp_in_tx['cm3_L']       = 0
    sp_in_tx['cm2_L']       = math.ceil(cm2_list[0]*64)
    sp_in_tx['cm1_L']       = math.ceil(-1*cm1_list[0]*64)
    sp_in_tx['c1_L']        = math.ceil(-1*c1_list[0]*64)
    sp_in_tx['c0_L']        = math.ceil(tx_swing*63)-sp_in_tx['cm3_L']-sp_in_tx['cm2_L']-sp_in_tx['cm1_L']-sp_in_tx['c1_L']
    sp_in_tx['pg_source ']  = 'ip'

    request = {'operation': 'function',
                   'function_name': f'set_pmd_e224g',
                   'params': sp_in_tx}
    evb.talk(request)

    #Store TX Results
    tx_res = {}
    opts={}
    tx_res['part_name']    = tx_part_name
    tx_res['sp']           = sp_in_tx
    tx_res['cal_res']      = evb.get_calibration_results(rx_lane, 'block', 'CMTX')
    # open the file in write mode
    tx_filename = tx_part_name + '_' + file_name + '_tx_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
    append_dict_to_csv(tx_filename, tx_res)

    '''=========================================================================
       RX Bring Up
       ========================================================================='''
    evb.select_dongle('RX')

    vr_evb = evb.eval(f"EVB_sm1b")
    evb.eval(
            f"""
            evb = EVB_sm1b;
            evb.write_vdd7_regulator('VOUT_CMD',{v['vpdig']});
            evb.set_vdd4_regulator({v['vdtx']});
            evb.set_vdd2_regulator({v['vprx']});
            evb.set_avdd2_regulator({v['vpcm']});
            evb.set_vdd6_regulator({v['avtt']});
            evb.set_vdd3_regulator({v['vphrx']});
            evb.set_avdd1_regulator({v['vphcm']});
            """
        )

    #Parameters for Logging Results
    xcount      = 0
    reset_iter  = 0
    temperature = rx_input_params['temperature']
    # open the file in write mode
    rx_filename = rx_part_name + '_' + file_name + '_rx_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'

    #Input arg dictionary
    input_arg = {}
    input_arg['key'] = {}
    disp_sp_in = {}
    input_arg['ext_data'] = v
    input_arg['save_pulse_resp']   = 1
    input_arg['process_adc_sram']  = 1
    for r in range(rx_input_params['nruns']):
        for cm1_i in cm1_list:
            if len(cm1_list) > 1:
                input_arg['key']['cm1'] = cm1_i
            for cm2_i in cm2_list:
                if len(cm2_list) > 1:
                    input_arg['key']['cm2'] = cm2_i
                for c1_i in c1_list:
                    if len(c1_list) > 1:
                        input_arg['key']['cm2'] = c1_i

                    evb.select_dongle('TX')

                    c1 = math.ceil(-1*c1_i*64)
                    cm1 = math.ceil(-1*cm1_i*64)
                    cm2 = math.ceil(cm2_i*64)
                    cm3 = 0
                    cursors = [cm3, cm2, cm1, 0, c1]
                    cursors[3] = math.ceil(tx_swing*63) - sum(cursors)

                    request = {'operation':'function',
                               'function_name':f'e224g_ip_cfg_set_txeq',
                               'params':[rx_lane, cursors, 'direct_codes', 1]}
                    evb.talk(request)

                    sp_args_rx['cm3_L'] = cm3
                    sp_args_rx['cm2_L'] = cm2
                    sp_args_rx['cm1_L'] = cm1
                    sp_args_rx['c1_L']  = c1
                    sp_args_rx['c0_L']  = cursors[4]

                    evb.select_dongle('RX')
                    for c in ctle_list:
                        if len(ctle_list) > 1:
                            input_arg['key']['ctle_boost'] = c
                        sp_args_rx['ctle_boost_L'] = c
                        for m in mf_list:
                            if len(mf_list) > 1:
                                input_arg['key']['ctle_mf'] = m
                            sp_args_rx['ctle_mf_L'] = m

                            for a in adpt_type_list:
                                if len(adpt_type_list) > 1:
                                    input_arg['key']['adapt_type'] = a
                                sp_args_rx['adapt_mode'] = a

                                match a:
                                    case 'pr': pra_list_live = pra_list
                                    case 'dfe': pra_list_live = float('nan')

                                for p in pra_list_live:
                                    if len(pra_list_live) > 1:
                                        input_arg['key']['pr'] = p
                                    sp_args_rx['pr_alpha'] = p

                                    reset_iter = reset_iter + 1 #Number of resets

                                    request = {'operation': 'function',
                                                'function_name': f'set_pmd_e224g',
                                                'params': sp_args_rx}
                                    sp_out = evb.talk(request)

                                    for read_iter in range(rx_input_params['nreads']):
                                        xcount = xcount + 1
                                        disp_sp_in['cm3'] = sp_args_rx['cm3_L']
                                        disp_sp_in['cm2'] = sp_args_rx['cm2_L']
                                        disp_sp_in['cm1'] = sp_args_rx['cm1_L']
                                        disp_sp_in['c0'] = sp_args_rx['c0_L']
                                        disp_sp_in['c1'] = sp_args_rx['c1_L']
                                        disp_sp_in['pr_alpha'] = sp_args_rx['pr_alpha']
                                        input_arg['sp_in']     = disp_sp_in

                                        ipid = evb.eval('get_pid(\'ip\')')

                                        request = {'operation': 'function',
                                                'function_name': f'e224g_rx_metrics',
                                                'params': [ipid, rx_lane, rx_filename, rx_part_name, xcount, 
                                                           reset_iter, read_iter, temperature, input_arg]}
                                        
                                        rx_res = evb.talk(request)
                                        append_dict_to_csv(rx_filename, rx_res)

                                        time.sleep(rx_input_params['dwell_time'])