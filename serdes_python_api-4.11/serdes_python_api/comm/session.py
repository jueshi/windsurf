"""Contains the Session class.
"""

import os
import time
from typing import Any

from serdes_python_api.comm.client import Client

REVISION = '$Revision: #2 $'
DATE = '$Date: 2024/05/09 $'


class Session:
    """Interacts with the server through a Client instance. This class interfaces with the user
    unlike Client
    """

    def __init__(self, host: str = 'localhost', port: int = 7878) -> None:
        """Creates and saves a Client instance

        :param host: The ip of the machine running the server, defaults to 'localhost'
        :param port: The port that the machine running the server is listening on, defaults to 7878
        """

        self.client = Client(host, port)

    @staticmethod
    def launch_server(exe_path: str, port: int) -> None:
        """Launch the API server from a Demo GUI executable, allowing users to fully automate the
        process of launching the server executable and running scripts

        :param exe_file_name: Absolute path to the compiled demo gui
        :raises FileNotFoundError: Executable not found
        :raises ConnectionAbortedError: Timeout
        :raises Exception: Other error occured
        """

        if not os.path.isfile(exe_path):
            raise FileNotFoundError(f"Could not find file '{exe_path}'")

        timeout_threshold = 90  # hard limit on connection attempt time

        no_connection = True
        start_attempt_time = time.time()
        curr_attempt_time = time.time()
        while no_connection:
            try:
                os.system(f"start {exe_path} api {port}")
                print(f"Starting API server on port {port}. "
                "Please wait, this may take a minute...\n")
                cstr = r"-------------------------------------\n"
                cstr += r"--- client connection established ---\n"
                cstr += r"-------------------------------------\n"
                # attempt connection
                print(cstr)
                no_connection = False
            except ConnectionRefusedError as e:
                # allow refused connections to pass
                time.sleep(1)
                curr_attempt_time = time.time()
                if (curr_attempt_time-start_attempt_time) > timeout_threshold:
                    raise ConnectionAbortedError("Taking too long to connect. " +
                                                 "Please retry and check port numbers...\n") from e
            except Exception as other_error:
                # raise any other exception type
                raise other_error
        # successfully connected
        print(f"Server connected and listening on port {port}\n")

    def exit(self) -> None:
        """Shuts the server down
        """

        print(self.client.talk('exit'))

    def eval(self, to_eval: str) -> Any:
        """Standard eval ran on the MATLAB side. Attempts to avoid failure

        :param to_eval: String to feed into the MATLAB eval function
        :return: Return from the MATLAB eval function
        """

        request = {'operation': 'eval',
                   'eval_string': to_eval}

        response = self.client.talk(request)
        return response

    def eval_return_all(self, to_eval: str) -> dict:
        """Standard eval ran on the MATLAB side. Attempts to avoid failure. Returns every variable
        created in the scope of the eval as a dictionary

        :param to_eval: String to feed into the MATLAB eval function
        :return: Dictionary containing all the variables declared during the course of the
            evaluation
        """

        request = {'operation': 'eval_return_all',
                   'eval_string': to_eval}

        response = self.client.talk(request)
        return response

    def run_script(self, path: str) -> dict:
        """Reads a MATLAB script from file and applies the eval_return_all function to it

        :param path: MATLAB script file name (or path if not in current directory)
        :return: Dictionary containing all the variables declared within the script
        """

        with open(path, 'r', encoding='utf-8') as file_stream:
            script_string = file_stream.read()
        script_vars = self.eval_return_all(script_string)
        return script_vars

    def init_ftd2xx(self, verbosity: int = 0) -> int:
        """Initializes the JTAG dongle. Can also be used to check the health of the JTAG dongle
        connection

        :param verbosity: Verbosity level of terminal output
        :return: 0 for success, non-zero otherwise
        """

        request = {'operation': 'function',
                   'function_name': 'init_ftd2xx',
                   'params_names': ['verbosity'],
                   'params': [verbosity]}

        response = self.client.talk(request)
        return response

    def read_id_univ(self) -> int:
        """Display hardware information in the command window

        :return: 0 for success
        """

        request = {'operation': 'function',
                   'function_name': 'read_id_univ',
                   'params_names': [],
                   'params': []}

        response = self.client.talk(request)
        return response
