"""Contains methods for plotting and measuring rx performance
"""

from serdes_python_api.e32g.e32g import E32G

REVISION = '$Revision: #3 $'
DATE = '$Date: 2024/02/02 $'


def eye(evb: E32G, lane: int) -> None:
    """Plot an eye diagram
    """

    # axes position given by https://www.mathworks.com/matlabcentral/answers/509445-why-uiaxes-does-
    # not-fill-uifigure-and-appears-small#answer_418905
    evb.eval(f"""
    fig = figure('NumberTitle', 'off');
    ax = uiaxes(fig);
    ax.YGrid = 'on';
    ax.YLabel.String = 'Amplitude [mV]';
    ax.XGrid = 'on';
    ax.XLabel.String = 'Time [UI]';
    ax.Position = [ax.Position(1:2) [fig.Position(3:4) - 2*ax.Position(1:2)]];
    e32g_rx_scope(get_pid('ip'), {lane}, struct('data_rate',
    part_params(get_pid('ip')).config.lane({lane}+1).settings.baudrate * 1e9,
    'phs_step', 1, 'dac_step', 1, 'scope_dac_range', 2, 'fig_composite', ax, 'safety_factor',
    2, 'dac_range', 656.25, 'stat_len', 32000, 'debug', 0, 'demo', 1, 'val_plot', 1));
    """.replace('\n', ''))


def bathtub(evb: E32G, lane: int, orientation: str, target_ber: float = -9.5, plot: int = 1,
            e15_meas: int = 0) -> dict:
    """Create a bathtub plot

    :param orientation: 'horizontal', 'vertical', or 'both' (horizontal + vertical)
    :param target_ber: Target BER value for bathtub calculation
    :param e15_meas: To return EH_15 and EW_15 bathtub parameters
    :param plot_fig: To plot bathtub graph
    """

    if orientation == 'horizontal':
        calc_eh = 0
        calc_ew = 1
    elif orientation == 'vertical':
        calc_eh = 1
        calc_ew = 0
    elif orientation == 'both':
        calc_eh = 1
        calc_ew = 1
    else:
        raise ValueError('orientation must be \'horizontal\' or \'vertical\' or \'both\'')

    return evb.eval_return_all(f"""
    result = e32g_bathtub_cal_eh_ew({lane}, struct('CALC_EH', {calc_eh}, 'CALC_EW', {calc_ew},
    'target_BER', {target_ber}, 'plot_fig', {plot}, 'E15_MEAS', {e15_meas}));
    """.replace('\n', ''))
