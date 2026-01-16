from curi_udp import curi_udp

import time
if __name__ == '__main__':
    CS = curi_udp("127.0.0.1", 10087, 10086)
    CS.open()
    for i in range(1000):
        data = CS.recieve()
        time.sleep(0.01)
        if data != "":
            print(data)
            CS.send(str(float(data) + 0.01))
    CS.close()

