import sys
import fcntl
import struct
import socket as s
import termios
import array, fcntl, struct, termios, os
import sys

from scapy.all import *

# How to send a packet
# (ifname, proto[, pkttype[, hatype[, addr]]])

def human_mac_to_bytes(addr):
    return bytes.fromhex(addr.replace(':', ''))

ifname='eth0'
dstmac='3D:91:57:CA:82:03'
payload='1234567890'
eth_type=b'\x7A\x05'

def create_frame(ifname, dstmac, eth_type, payload):
    
    print(os.getpgrp())
    info = fcntl.ioctl(1, termios.TIOCGPGRP, struct.pack('256s', bytes('wlo1', 'utf-8')[:15]), 1)
    srcmac = ':'.join('%02x' % b for b in info[18:24])

    # Build Ethernet frame
    payload_bytes = payload.encode('utf-8')
    assert len(payload_bytes) <= 1500  # Ethernet MTU

    frame = human_mac_to_bytes(dstmac) + \
            human_mac_to_bytes(srcmac) + \
            eth_type + \
            payload_bytes

    # Create Ethernet frame
    return frame

def create_packet(dstaddress, udp_port):
    return IP(dst=dstaddress)/UDP(dport=udp_port)

