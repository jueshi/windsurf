"""Contains the Client class
"""

import json
import socket
from functools import partial
from typing import Any

REVISION = '$Revision: #2 $'
DATE = '$Date: 2024/01/10 $'


class Client:
    """Manages communication between the server and Python
    """

    def __init__(self, host: str = 'localhost', port: int = 7878) -> None:
        """Save host and port for making connections

        :param host: The ip of the machine running the server, defaults to 'localhost'
        :param port: The port that the machine running the server is listening on, defaults to 7878
        """

        self.host = host
        self.port = port

    def talk(self, request: str | dict) -> Any:
        """Connects to IP <host> on port <port> and transmits a byte string request, then waits for
        a response. After the response is received, the connection is terminated to prevent socket
        lock up

        :param request: A dictionary in the form of the expected API server input structure, or
            the string 'exit' to shut down the server application
        :return: Data requested with the api call
        """

        if request != 'exit':  # exit message can be sent as raw text
            request = json.dumps(request)

        # blocking non-persistent connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))  # connect to server
            byte_message_to_send = bytes(f'{request}', 'utf-8')
            s.sendall(byte_message_to_send)  # send message to server

            # \\\\ SERVER PROCESSING REQUEST //// #

            # retrieve response message from server
            raw_data_list = []
            for data in iter(partial(s.recv, 4096), b''):
                raw_data_list.append(data.decode())

            # final response message
            raw_data = ''.join(raw_data_list)

        # return processed json
        return self.manage_return(raw_data)

    def manage_return(self, raw_data: str) -> Any:
        """Decodes the incoming JSON data. Raises an exception if there was an error on the server
        side and prints the error message sent over by the server

        :param raw_data: Raw JSON encoded byte stream as string
        :raises RuntimeError: Error raised by the server during execution of the string
        :return: Data requested with the API call
        """

        structured_data = json.loads(raw_data)

        exception_status = structured_data['MLError']
        return_data = structured_data['data']

        if exception_status:  # server side error, print error message and raise exception
            error_msg = return_data['message']
            if return_data['identifier']:
                error_msg = return_data['identifier'] + ': \n\n' + error_msg
            raise RuntimeError(error_msg)

        # no server side error, return just data
        return return_data
