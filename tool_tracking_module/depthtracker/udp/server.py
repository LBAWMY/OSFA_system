from curi_udp import curi_udp
import time

if __name__ == '__main__':
    CS = curi_udp("127.0.0.1", 10086, 10087)
    CS.open()
    for i in range(1000):
        now = time.time()
        CS.send(str(now))
        time.sleep(0.01)
        data = CS.recieve()
        if data != "":
            print("diff time ", float(data) - now)
    CS.close()
