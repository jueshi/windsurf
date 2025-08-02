"""Contains the Shared class
"""

import json
from datetime import datetime
from typing import Any, List, Literal

from serdes_python_api.comm.session import Session

REVISION = '$Revision: #16 $'
DATE = '$Date: 2025/01/17 $'


class Shared():
    """Contains product generic methods related to configuration and evaluation
    """

    def __init__(self, host: str = 'localhost', port: int = 7878) -> None:
        """Create placeholders for needed attributes
        """
        self.session = Session(host, port)
        self.sp_args = self._fetch_set_part_args()
        self.protocol_defaults = self._generate_protocol_defaults(self.sp_args['standard_L'])
        self.clock_args = {'freqMHz': self.protocol_defaults['refclk']}
        self.product = self._generate_product_string()
        self.ipid = self.eval('get_pid(\'ip\')')
        self.tpid = self.eval('get_pid(\'tc\')')
        self.part_params = self._get_part_params()

    def _set_clock(self) -> None:
        request = {'operation': 'function',
                   'function_name': 'config_clock',
                   'params': self.clock_args}
        self.talk(request)

    def _set_part(self) -> dict:
        # Get set_part script
        product = (self.part_params[self.ipid-1]['product']).lower()

        match product:
            case "e32g" | "c20g2" | "c20g":
                script = 'set_part_' + product
            case "e112mp":
                gen = self.part_params[self.ipid-1]['gen']
                script = 'set_pmd_' + product + '_' + gen
            case "pcie6" | "pcie6_v2" | "pcie6_lf" | "e224g" | 'vsr':
                script = 'set_pmd_' + product

        request = {'operation': 'function',
                   'function_name': f'{script}',
                   'params': self.sp_args}
        sp_out = self.talk(request)
        return sp_out

    def _fetch_set_part_args(self) -> dict:
        with open('sp_args.json', encoding='utf-8') as file:
            return json.load(file)

    def _generate_product_string(self) -> str:
        return str(self).split('.', maxsplit=2)[1]

    def _generate_protocol_defaults(self, standard: str):
        return self.eval(f"get_protocol_defaults(get_pid('ip'), '{standard}')")

    def _get_part_params(self):
        return self.eval(f"part_params()")

    def sync_pm(self, lane: int) -> None:
        """Syncs the pattern matcher

        :param lane: Data path
        """

        self.eval(f"ip_cfg_sync_pm({lane})")

    def adapt(self, lane: int) -> None:
        """Runs adaptation

        :param lane: Data path
        """

        self.eval(f"rx_startup_adaptation(get_pid('ip'), {lane})")
        self.sync_pm(lane)

    def calibrate(self, lane: int) -> None:
        """Runs calibration

        param lane: Data path
        """

        self.eval(f"rx_startup_calibration(get_pid('ip'), {lane})")
        self.sync_pm(lane)

    def count_error(self, lane: int) -> int:
        """Returns the number of errors since the last sync_pm

        :param lane: Data path
        :return: Number of errors
        """

        return self.eval(f"ip_cfg_count_error({lane})")

    def inject_error(self, lane: int, errors: int) -> None:
        """Injects a specified number of errors

        :param lane: Data path
        :param errors: Number of errors to inject
        """

        self.eval(f"ip_cfg_inject_error({lane}, {errors}, struct())")

    def invert_polarity(self, lane: int, rx_tx: Literal['RX', 'TX'] = 'RX') -> None:
        """Invert the polarity

        :param lane: Data path
        :param rx_tx: 'RX' or 'TX'
        """

        self.eval(f"invert_polarity('{rx_tx}', {lane})")

    def set_pattern(self, lane: int, pattern: str, tx_rx: Literal['RX', 'TX', 'TXRX'] = 'TX') -> None:
        """Set the pattern

        :param lane: Data path
        :param pattern: Pattern to set
        :param tx_rx: 'TX', 'RX', or 'TXRX'
        """
        
        self.eval(f"ip_cfg_set_pattern({lane}, '{pattern}', '{tx_rx}')")

    def asr(self, pid: int, reg_name: str, value: int, rb_en: int = 0) -> bool | None:
        """Sets a register to a value

        :param pid: ID in the JTAG chain of the device
        :param reg_name: Name of the register to set a desired value
        :param value: Value to set the register
        :param rb_en: Readback the set value
        :return: Readback status or None
        """

        request = {'operation': 'asr',
                   'params': [pid, reg_name, value, rb_en]}
        response = self.talk(request)
        return response

    def agr(self, pid: int, reg_name: str, num_reads: int = 1) -> int | List[int]:
        """Gets the current value of a register

        :param pid: ID in the JTAG chain of the device
        :param reg_name: Name of the register to set a desired value
        :param num_reads: Number of times that register should be read
        :return: Integer value if a single read or list of integers if multiple reads
        """

        request = {'operation': 'agr',
                   'params': [pid, reg_name, num_reads]}
        response = self.talk(request)
        return response

    def reg_dump(self, pid: int) -> List[tuple]:
        """Return all registers belonging to the device

        :param pid: ID in the JTAG chain of the device
        :return: List of all registers
        """

        request = {'operation': 'function',
                   'function_name': 'reg_dump',
                   'params': [pid]}
        response = self.talk(request)

        # response is flattened, change shape to two columns
        num_fields = len(response)//2
        field_names = response[:num_fields]
        field_values = response[num_fields:]

        # return as list of tuples
        response = list(zip(field_names, field_values))
        return response

    def run_script(self, filename: str) -> None:
        """Run a MATLAB .m file

        :param filename: Name of the script to run
        """

        request = {'operation': 'function',
                   'function_name': 'run_script',
                   'params': filename}
        self.talk(request)

    def close_figures(self, save: bool = False) -> None:
        """Close all MATLAB figures

        :param save: Save figures to .png before closing
        """

        self.eval(
            f"""
            fig_list = findobj(allchild(0), 'flat', 'Type', 'figure');

            fig_names = strings(length(fig_list), 1);
            for i = 1:length(fig_list)
                fig_names(i) = string(fig_list(i).Name);
            end

            dels = ~contains(fig_names, ["Demo GUI","Tool Launcher"]);
            parsed_list = fig_list(dels);

            for i = 1:length(parsed_list)
                if {int(save)}
                set(0, 'CurrentFigure', parsed_list(i));
                mkdir([pwd '\\figures'])
                saveas(parsed_list(i), fullfile([pwd '\\figures'], ...
                ['{datetime.timestamp(datetime.now())}' num2str(i) '.png']));
                end
                close(parsed_list(i));
                delete(parsed_list(i));
            end
            """
        )

    def talk(self, request: str | dict) -> Any:
        """Wrapper for client's talk method

        :param request: A dictionary in the form of the expected API server input structure, or
            the string 'exit' to shut down the server application
        :return: Data requested with the api call
        """

        return self.session.client.talk(request)

    def eval(self, to_eval: str) -> Any:
        """Wrapper for session's eval method

        :param to_eval: String to feed into the MATLAB eval function
        :return: Return from the MATLAB eval function
        """

        return self.session.eval(to_eval)

    def eval_return_all(self, to_eval: str) -> dict:
        """Wrapper for session's eval_return_all method

        :param to_eval: String to feed into the MATLAB eval function
        :return: Dictionary containing all the variables declared during the course of the
            evaluation
        """

        return self.session.eval_return_all(to_eval)

    def launch_server(self, exe_file_name: str) -> None:
        """Wrapper for session's launch_server method

        :param exe_file_name: Absolute path to the compiled demo gui
        :raises FileNotFoundError: Executable not found
        :raises ConnectionAbortedError: Timeout
        :raises Exception: Other error occured
        """

        return self.session.launch_server(exe_file_name)

    def dict_to_mstruct(self, d: Any) -> str:
        """Convert Python dictionary to MATLAB struct (in string form)

        :param d: Dictionary to convert
        :return: String to evaluate
        """

        if not isinstance(d, dict):
            if d is None:
                return '[]'
            if isinstance(d, str):
                return f"'{d}'"
            return f'{d}'
        else:
            s = 'struct('
            for k, v in d.items():
                s += f"'{k}', {self.dict_to_mstruct(v)}"
            return s + ')'

    def func(self, function_name: str, *args) -> Any:
        """Call MATLAB function

        :param function_name: Name of function
        :param args: Arguments
        :return: Function return
        """

        return self.talk({'operation': 'function', 'function_name': function_name, 'params': args})
