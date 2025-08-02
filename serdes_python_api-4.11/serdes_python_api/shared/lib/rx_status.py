"""Module containing functions related to collecting/plotting rx data. Supports pcie6, e112mp, and
some of pcie6_v2
"""

import csv
from math import nan
from typing import Literal

from serdes_python_api.shared.shared import Shared

REVISION = '$Revision: #5 $'
DATE = '$Date: 2025/01/06 $'

SRAM_STRS = {'e112mp': {'adc': 'adc', 'edfe': 'emulated_dfe', 'ffe': 'cdr_ffe.ffe'},
             'pcie6': {'adc': 'adc_dfe.adc', 'edfe': 'emulated_dfe', 'ffe': 'cdr_ffe.ffe'},
             'pcie6_v2': {'adc': 'adc_no_pack'},
             'pcie6_lf': {'adc': 'adc_only', 'edfe': 'lms', 'ffe': 'cdr.ffe'},
             'e224g': {'adc': 'adc', 'edfe': 'edfe_pack', 'ffe': 'cdr_ffe.ffe'}}


def collect_edfe_vals(evb: Shared, lane: int, to_csv: bool = False) -> dict:
    """Collect eDFE data

    :param to_csv: Send data to csv file, defaults to False
    :return: eDFE data
    """

    if evb.product == 'e224g':
        vals = evb.eval_return_all(f"""
        c = rx_data_collector_{evb.product}({lane});
        c.capture_data_sram('{SRAM_STRS[evb.product]['edfe']}');
        edfe_vals = c.data_sram.emulated_dfe;
        """)

        if to_csv:
            with open('edfe_vals.csv', 'w', encoding='UTF-8') as file:
                writer = csv.writer(file)
                writer.writerow(vals['edfe_vals'])
    else:
        vals = evb.eval_return_all(f"""
        c = rx_data_collector_{evb.product}({lane});
        c.capture_data_sram('{SRAM_STRS[evb.product]['edfe']}');
        edfe1_vals = c.data_sram.emulated_dfe(1).val;
        edfe2_vals = c.data_sram.emulated_dfe(2).val;
        edfe3_vals = c.data_sram.emulated_dfe(3).val;
        """)

        if to_csv:
            with open('edfe_vals.csv', 'w', encoding='UTF-8') as file:
                writer = csv.writer(file)
                writer.writerow(vals['edfe1_vals'])
                writer.writerow(vals['edfe2_vals'])
                writer.writerow(vals['edfe3_vals'])
    return vals


def collect_adc_vals(evb: Shared, lane: int, to_csv: bool = False) -> dict:
    """Collect ADC data

    :param to_csv: Send data to csv file, defaults to False
    :return: ADC data
    """

    vals = evb.eval_return_all(f"""
    c = rx_data_collector_{evb.product}({lane});
    c.capture_data_sram('{SRAM_STRS[evb.product]['adc']}');
    adc_vals = c.data_sram.adc_dfe;
    """)

    if to_csv:
        with open('adc_vals.csv', 'w', encoding='UTF-8') as file:
            writer = csv.writer(file)
            writer.writerow(vals['adc_vals'])
    return vals


def plot_edfe_eye(evb: Shared, lane: int) -> None:
    """Create an eye diagram plot using eDFE SRAM data
    """

    evb.eval(f"""
    c = rx_data_collector_{evb.product}({lane});
    c.capture_data_sram('{SRAM_STRS[evb.product]['edfe']}');
    d = rx_data_drawer_{evb.product}(c);
    d.draw_graph_edfe_dsp_eye();
    f = copyobj(d.graph_edfe_dsp_eye, figure('Name', 'eDFE Eye'));
    drawnow;
    """)


def plot_bathtub(evb: Shared, lane: int, ber_level: float = nan) -> list[int]:
    """Create a bathtub plot using edfe sram data

    :param ber_level: BER to measure eye margin at
    :return: margins at ber_level if not nan
    """

    r = evb.eval_return_all(f"""
    c = rx_data_collector_{evb.product}({lane});
    c.capture_data_sram('{SRAM_STRS[evb.product]['edfe']}');
    c.capture_data_adaptation();
    d = rx_data_drawer_{evb.product}(c);
    margins = d.draw_graph_bathtub({ber_level});
    f = copyobj(d.graph_bathtub, figure('Name', 'Bathtub'));
    drawnow;
    """)
    return r['margins']


def plot_adc_histo_scatter(evb: Shared, lane: int,
                           histo_only: Literal['true', 'false'] = 'false') -> None:
    """Create an adc histo scatter plot

    "param histo_only: Plot only the histogram portion of the histo scatter
    """

    evb.eval(f"""
    c = rx_data_collector_{evb.product}({lane});
    c.capture_data_sram('{SRAM_STRS[evb.product]['adc']}');
    c.capture_data_sram('{SRAM_STRS[evb.product]['edfe']}');
    d = rx_data_drawer_{evb.product}(c);
    in.histo_only = {histo_only};
    d.draw_graph_adc_histo_scatter(in);
    f = copyobj(d.graph_adc_histo_scatter, figure('Name', 'ADC Histo Scatter'));
    drawnow;
    """)


def plot_edfe_histo_scatter(evb: Shared, lane: int,
                            histo_only: Literal['true', 'false'] = 'false') -> None:
    """Create an edfe histo scatter plot

    "param histo_only: Plot only the histogram portion of the histo scatter
    """

    evb.eval(f"""
    c = rx_data_collector_{evb.product}({lane});
    c.capture_data_adaptation();
    c.capture_data_sram('{SRAM_STRS[evb.product]['edfe']}');
    d = rx_data_drawer_{evb.product}(c);
    in.histo_only = {histo_only};
    d.draw_graph_edfe_histo_scatter(in);
    f = copyobj(d.graph_edfe_histo_scatter, figure('Name', 'eDFE Histo Scatter'));
    drawnow;
    """)
