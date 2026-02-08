"""Module contains the implementation of the Bert class.
"""

from typing import List, Union
from serdes_python_api.pcie6.evb import Evb

REVISION = '$Revision: #1 $'
DATE = '$Date: 2024/01/04 $'


class Bert():
    """Responsible for controlling a connected BERT.
    """

    def __init__(self, connected=True) -> None:
        """Create Bert instance.

        :param connected: BERT is connected to MATLAB backend, defaults to True
        """

        self.evb = Evb()
        self.connected = connected
        if self.connected:
            self.evb.eval("manage_instruments('init')")

    def set_baudhz(self, baudhz: int) -> None:
        """Set the baud rate.

        :param baudhz: Baud rate (Hz)
        """

        if not self.connected:
            input(f"Set the baudHz to {baudhz:.1e}\n"
                  r"> Press any key to continue")
            return

        self.evb.eval(f"set_bert('baudHz={baudhz}')")

    def set_amplitude(self, kind: str, amp: Union[int, float]) -> None:
        """Set the amplitude.

        :param kind: 'PER' or 'SIN'
        :param amp: Amplitude (UI)
        """

        if not self.connected:
            input(f"Set the amplitude to {amp:.2f} with type {kind}\n"
                  r"> Press any key to continue")
            return

        self.evb.eval(f"set_bert('PJstate=on,PJtype={kind},PJampUI={amp}')")

    def set_frequency(self, kind: str, freq: int) -> None:
        """Set the frequency.

        :param kind: 'PER' or 'SIN'
        :param freq: Frequency (hz)
        """

        if not self.connected:
            input(f"Set the frequency to {freq:.0e} with type {kind}\n"
                  r"> Press any key to continue")
            return

        self.evb.eval(f"set_bert('PJstate=on,PJtype={kind},PJfreqHz={freq}')")

    def set_ssc(self, standard: str) -> None:
        """Set SSC.

        :param standard: Configured standard
        """

        ssc = int(standard.endswith('_SRIS'))
        if not self.connected:
            input(f"Set SSC to {ssc}\n"
                  r"> Press any key to continue")
            return

        self.evb.eval("manage_instruments('Mbert',':SOURce:SSCLocking:GLOBal:"
                      f"STATe  \"M1.System\",{ssc};:STATus:INSTrument:RUN:WAIT?"
                      " \"M1.DataOut1\"');")

    def toggle_data(self, cmd: str) -> None:
        """Toggle data on/off

        :param cmd: 'on' or 'off'
        """

        if not self.connected:
            input(f"Set data to {cmd.upper()}\n"
                  r"> Press any key to continue")
            return

        if cmd == 'on':
            self.evb.eval("set_bert('globaldata=on')")
        elif cmd == 'off':
            self.evb.eval("set_bert('globaldata=off')")

    def get_max_amplitude(self, kind: str) -> int:
        """Get the maximum aplitude set.

        :param type: 'PER' or 'SIN'
        :return: Amplitude (UI)
        """

        if not self.connected:
            try:
                out = getattr(self, f'max_{kind}_amp')
            except AttributeError:
                out = float(input(rf"> Enter bert max amp for type {kind}: "))
                setattr(self, f'max_{kind}_amp', out)
            return out

        return self.evb.eval(f"determine_max_sj('{kind}')")

    def get_tx_eq(self) -> List[float]:
        """Return a list of the tx eq cursors from the BERT.

        :return: Dictionary with 3 cursors
        """

        if not self.connected:
            return input(r"> Enter TX EQ cursors as a list: ")

        cursors = [0, 1, 3]
        return {i: self.evb.eval("manage_instruments('Mbert',':OUTPut:DEEMphasis"
                                 f":CURSor:MAGNitude{i}? ''M1.DataOut1''')")
                .partition('\n')[0]
                for i in cursors}
