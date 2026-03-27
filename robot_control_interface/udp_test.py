import socket
import time

# local 192.168.4.4 8000
# exe: 192.168.4.4 8000 192.168.3.3 8001

HOST = '192.168.10.1'    # local ip and port
PORT = 8000
addr = (HOST, PORT)
UDPSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# UDPSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
UDPSock.bind((HOST, PORT))

print('...Waiting for message...')

Robot_addr_send = ('192.168.10.10', 8001)    # remote ip and port

# UDOSocket_data, UDOSocket_client_add = UDPSock.recvfrom(1024)
# print('...recieve message !!!')
# print('... udp ... str(self.UDOSocket_data) + str(self.UDOSocket_client_add).....')
#
# udp_save = open("udp_save.txt", 'w')
# udp_save.write(str(UDOSocket_data) + str(UDOSocket_client_add))
# print(str(UDOSocket_data) + str(UDOSocket_client_add))

while True:
    # print('receive_data')
    send_data_ls = [0,0,0,0,0,0,0,10]
    UDPSock.sendto(bytearray(send_data_ls), Robot_addr_send)
    print('send_data', send_data_ls)
    # UDOSocket_data, UDOSocket_client_add = UDPSock.recvfrom(1024)
    # UDOSocket_data = UDPSock.recvfrom(1024)[0]
    UDOSocket_data = UDPSock.recvfrom(1024)[0]
    # UDPSock.sendto(bytearray([0,0,0,0,0,0,0,0]), UDOSocket_client_add)
    # print('UDOSocket_data', UDOSocket_data, UDOSocket_client_add)
    # z, pitch. yaw, insertion, grasp, tilt
    # print(UDOSocket_data)
    print('receive_data', UDOSocket_data[0], UDOSocket_data[1], UDOSocket_data[2],
          UDOSocket_data[3], UDOSocket_data[4], UDOSocket_data[5])
    # print(UDPSock.recvfrom(1024)[0])
    # print(int.from_bytes(UDOSocket_data, byteorder='little', signed=True))

UDPSock.close()
