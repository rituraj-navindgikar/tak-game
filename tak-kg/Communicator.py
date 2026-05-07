import socket, sys
from subprocess import Popen, PIPE
from nbstreamreader import NonBlockingStreamReader as NBSR
from sys import platform
import os


class Communicator(object):
    def __init__(self):
        self.Socket = None
        self.ChildProcess = None
        self.ModifiedOutStream = None

    def setSocket(self, Socket, TIMEOUT=60):
        self.Socket = Socket
        self.Socket.settimeout(TIMEOUT)

    def isSocketNotNone(self):
        return self.Socket is not None

    def isChildProcessNotNone(self):
        return self.ChildProcess is not None

    def closeSocket(self):
        if self.isSocketNotNone():
            try:
                self.Socket.close()
            except Exception:
                pass
            self.Socket = None

    def SendDataOnSocket(self, data):
        success_flag = False
        if self.isSocketNotNone():
            try:
                if isinstance(data, str):
                    data = data.encode("utf-8")   # str -> bytes for Python 3 sockets
                self.Socket.sendall(data)
                success_flag = True
            except Exception:
                pass
        return success_flag

    def RecvDataOnSocket(self):
        """Returns raw bytes; caller decodes if needed."""
        data = None
        if self.isSocketNotNone():
            while True:
                try:
                    data = self.Socket.recv(1024)
                except Exception:
                    data = None
                    break
                if data is None:
                    break
                elif len(data) > 0:
                    break
        return data

    def CreateChildProcess(self, Execution_Command, Executable_File):
        if platform in ("darwin", "linux", "linux2"):
            self.ChildProcess = Popen(
                [Execution_Command, Executable_File],
                stdin=PIPE,
                stdout=PIPE,
                bufsize=0,
                preexec_fn=os.setsid
            )
        else:
            self.ChildProcess = Popen(
                [Execution_Command, Executable_File],
                stdin=PIPE,
                stdout=PIPE,
                bufsize=0
            )
        self.ModifiedOutStream = NBSR(self.ChildProcess.stdout)

    def RecvDataOnPipe(self, TIMEOUT):
        """Returns str (decoded). Returns None on timeout."""
        data = None
        if self.isChildProcessNotNone():
            try:
                data = self.ModifiedOutStream.readline(TIMEOUT)
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8", errors="replace")
            except Exception:
                pass
        return data

    def SendDataOnPipe(self, data):
        """Accepts str or bytes. Encodes str before writing to stdin pipe."""
        success_flag = False
        if self.isChildProcessNotNone():
            try:
                if isinstance(data, str):
                    data = data.encode("utf-8")   # Popen stdin in binary mode expects bytes
                self.ChildProcess.stdin.write(data)
                self.ChildProcess.stdin.flush()
                success_flag = True
            except Exception:
                pass
        return success_flag

    def closeChildProcess(self):
        if self.isChildProcessNotNone():
            if platform in ("darwin", "linux", "linux2"):
                try:
                    os.killpg(os.getpgid(self.ChildProcess.pid), 15)
                except Exception:
                    pass
            else:
                try:
                    self.ChildProcess.kill()
                except Exception:
                    pass
            self.ChildProcess = None


if __name__ == '__main__':
    c = Communicator()
    c.CreateChildProcess('sh', 'run.sh')
    counter = 1
    try:
        while counter != 100:
            c.SendDataOnPipe(str(counter) + '\n')
            data = c.RecvDataOnPipe(1)
            print('Parent Received', data)   # print statement -> function
            if data is None:
                continue
            data = data.strip()
            counter = int(data)
    except Exception:
        c.closeChildProcess()