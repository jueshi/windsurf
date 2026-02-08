"""Module containing functions related to collecting/plotting rx data for pcie6_v2
"""

import csv
from math import nan
from typing import Literal

from serdes_python_api.shared.shared import Shared

REVISION = '$Revision: #5 $'
DATE = '$Date: 2024/06/27 $'


def get_calibration_results(evb: Shared, lane: int) -> dict:
    """Retrieve calibration results
    """

    return evb.eval_return_all(f"""
    c = rx_data_collector_pcie6_v2({lane})
    c.capture_data_calibration();
    res = c.data_calibration;
    """)['res']


def get_adaptation_results(evb: Shared, lane: int) -> dict:
    """Retrieve adaptation results
    """

    return evb.eval_return_all(f"""
    c = rx_data_collector_pcie6_v2({lane})
    c.capture_data_adaptation();
    res = c.data_adaptation;
    """)['res']


def collect_edfe_norm_vals(evb: Shared, lane: int, to_csv: bool = False) -> dict:
    """Collect normalized eDFE data

    :param to_csv: Send data to csv file, defaults to False
    :return: eDFE data
    """

    vals = evb.eval_return_all(f"""
    c = rx_data_collector_pcie6_v2({lane});
    c.capture_data_sram_edfe_norm();
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


def plot_edfe_norm_eye(evb: Shared, lane: int) -> None:
    """Create an eye diagram plot using normalized eDFE data
    """

    evb.eval(f"""
    c = rx_data_collector_pcie6_v2({lane});
    c.capture_data_sram_edfe_norm();
    d = rx_data_drawer_pcie6_v2(c);
    d.draw_graph_edfe_dsp_eye();
    screen = get(0, 'ScreenSize');
    pos = [screen(3)/2 - 280, screen(4)/2 - 210, 560, 420];
    f = copyobj(d.graph_edfe_dsp_eye, figure('Name', 'eDFE Eye', 'Position', pos));
    drawnow;
    """)


def plot_bathtub(evb: Shared, lane: int, ber_level: float = nan) -> list[int]:
    """Create a bathtub plot using edfe sram data

    :param ber_level: BER to measure eye margin at
    """

    r = evb.eval_return_all(f"""
    c = rx_data_collector_pcie6_v2({lane});
    c.capture_data_sram_edfe_norm();
    d = rx_data_drawer_pcie6_v2(c);
    margins = d.draw_graph_bathtub({ber_level});
    screen = get(0, 'ScreenSize');
    pos = [screen(3)/2 - 280, screen(4)/2 - 210, 560, 420];
    f = copyobj(d.graph_bathtub, figure('Name', 'Bathtub', 'Position', pos));
    drawnow;
    """)
    return r['margins']


def plot_adc_histo_scatter(evb: Shared, lane: int,
                           histo_only: Literal['true', 'false'] = 'false') -> None:
    """Create an adc histo scatter plot

    param histo_only: Plot only the histogram portion of the histo scatter
    """

    evb.eval(f"""
    c = rx_data_collector_pcie6_v2({lane});
    c.capture_data_sram('adc_no_pack');
    c.capture_data_sram_edfe_norm();
    d = rx_data_drawer_pcie6_v2(c);
    in.histo_only = {histo_only};
    in.ylim = 33;
    d.draw_graph_adc_histo_scatter(in);
    screen = get(0, 'ScreenSize');
    pos = [screen(3)/2 - 280, screen(4)/2 - 210, 560, 420];
    f = copyobj(d.graph_adc_histo_scatter, figure('Name', 'ADC Histo Scatter', 'Position', pos));
    drawnow;
    """)


def plot_edfe_histo_scatter(evb: Shared, lane: int,
                            histo_only: Literal['true', 'false'] = 'false') -> None:
    """Create an edfe histo scatter plot

    param histo_only: Plot only the histogram portion of the histo scatter
    """

    evb.eval(f"""
    c = rx_data_collector_pcie6_v2({lane});
    c.capture_data_adaptation();
    c.capture_data_sram_edfe_norm();
    d = rx_data_drawer_pcie6_v2(c);
    in.histo_only = {histo_only};
    d.draw_graph_edfe_histo_scatter(in);
    screen = get(0, 'ScreenSize');
    pos = [screen(3)/2 - 280, screen(4)/2 - 210, 560, 420];
    f = copyobj(d.graph_edfe_histo_scatter, figure('Name', 'eDFE Histo Scatter', 'Position', pos));
    drawnow;
    """)
