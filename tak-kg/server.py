import socket
import sys
import json
import pdb
from Communicator import Communicator
import argparse


class Server:
    def __init__(self):
        self.communicator_list = []
        self.NETWORK_TIMER = 500

    def BuildServer(self, port_no, num_clients):
        s = socket.socket()
        s.settimeout(self.NETWORK_TIMER)
        host = "0.0.0.0"
        self.port = port_no
        s.bind((host, port_no))
        s.listen(5)
        self.client_count = 0
        self.CLOSE_NETWORK = False

        while self.client_count < num_clients and (not self.CLOSE_NETWORK):
            try:
                c, addr = s.accept()
            except socket.timeout:
                self.CLOSE_NETWORK = True
                break
            except Exception:
                self.CLOSE_NETWORK = True
                break

            if not self.CLOSE_NETWORK:
                self.client_count += 1
                self.communicator_list.append(Communicator())
                self.communicator_list[-1].setSocket(c, self.NETWORK_TIMER)

        s.close()

    def setNetworkTimer(self, Time_in_seconds):
        self.NETWORK_TIMER = Time_in_seconds

    def getNetworkTimer(self):
        return self.NETWORK_TIMER

    def RecvDataFromClient(self, client_id):
        data = None
        if client_id < len(self.communicator_list):
            data = self.communicator_list[client_id].RecvDataOnSocket()
            if data is None:
                print('ERROR : TIMEOUT ON CLIENT NETWORK ' + str(client_id) + ' END')
                self.CloseClient(client_id)
        return data

    def SendData2Client(self, client_id, data):
        success_flag = False
        if data is None:
            data = {'meta': 'TIMEOUT ON CLIENT NETWORK', 'action': 'KILLPROC', 'data': ''}
        else:
            # data arrives as bytes from socket; decode before json.loads
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", errors="replace")
            data = json.loads(data)

        if client_id < len(self.communicator_list):
            success_flag = self.communicator_list[client_id].SendDataOnSocket(json.dumps(data))
            if not success_flag:
                print('ERROR : COULD NOT SEND DATA TO CLIENT ' + str(client_id))
                self.CloseClient(client_id)
            elif (data['action'] == 'KILLPROC') or (data['action'] == 'FINISH'):
                self.CloseClient(client_id)
        return success_flag

    def CloseClient(self, client_id):
        if client_id < len(self.communicator_list):
            self.communicator_list[client_id] = None

    def CloseAllClients(self):
        for idx in range(len(self.communicator_list)):
            if self.communicator_list[idx] is not None:
                self.CloseClient(idx)
        self.communicator_list = []

    def SendInitError2Clients(self):
        for idx in range(len(self.communicator_list)):
            if self.communicator_list[idx] is not None:
                data = {'meta': 'ERROR IN INITIALIZATION', 'action': 'KILLPROC', 'data': ''}
                self.SendData2Client(idx, json.dumps(data))
                self.CloseClient(idx)

    def playTak(self, n, timelimit, client_0, client_1):
        if (client_0 < len(self.communicator_list)) and (client_1 < len(self.communicator_list)):
            dataString = '1 ' + str(n) + ' ' + str(timelimit)
            data = {'meta': '', 'action': 'INIT', 'data': dataString}
            self.SendData2Client(client_0, json.dumps(data))
            data['data'] = '2 ' + str(n) + ' ' + str(timelimit)
            self.SendData2Client(client_1, json.dumps(data))

            while True:
                data = self.RecvDataFromClient(client_0)
                # socket gives bytes in Python 3; decode before forwarding/parsing
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8", errors="replace")

                self.SendData2Client(client_1, data)
                if not data:
                    break
                print(data, 'Received from client 0')
                parsed = json.loads(data)
                if parsed['action'] == 'FINISH' or parsed['action'] == 'KILLPROC':
                    break

                data = self.RecvDataFromClient(client_1)
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8", errors="replace")
                print(data, 'Received from client 1')
                self.SendData2Client(client_0, data)
                if not data:
                    break
                parsed = json.loads(data)
                if parsed['action'] == 'FINISH' or parsed['action'] == 'KILLPROC':
                    break

            self.CloseClient(client_0)
            self.CloseClient(client_1)
        else:
            self.CloseAllClients()


if __name__ == '__main__':
    print('Start')
    local_Server = Server()
    parser = argparse.ArgumentParser(description='Tak Server')
    parser.add_argument('port', metavar='10000', type=int, help='Server port')
    parser.add_argument('-n', dest='n', metavar='N', type=int, default=5, help='Tak board size')
    parser.add_argument('-NC', dest='num_clients', metavar='num_clients', type=int, default=2,
                        help='Number of clients connecting to the server')
    parser.add_argument('-TL', dest='time_limit', metavar='time_limit', type=int, default=120,
                        help='Time limit (in s)')
    args = parser.parse_args()

    print('Waiting for clients to connect...')
    local_Server.BuildServer(args.port, args.num_clients)

    if local_Server.client_count < 2:
        local_Server.SendInitError2Clients()
        print('Not enough clients connected to start the game.')
    else:
        local_Server.playTak(args.n, args.time_limit, 0, 1)
        print('Game Finished')