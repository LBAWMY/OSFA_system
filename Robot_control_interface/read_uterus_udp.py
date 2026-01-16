import socket

HOST = '192.168.10.1'
PORT = 8000
addr = (HOST, PORT)
UDPSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# UDPSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
UDPSock.bind((HOST, PORT))

print('...Waiting for message...')
UDOSocket_data, UDOSocket_client_add = UDPSock.recvfrom(1024)
print('...recieve message !!!')
print('... udp ... str(self.UDOSocket_data) + str(self.UDOSocket_client_add).....')

udp_save = open("udp_save.txt", 'w')
udp_save.write(str(UDOSocket_data) + str(UDOSocket_client_add))
print(str(UDOSocket_data) + str(UDOSocket_client_add))

UDPSock.close()

