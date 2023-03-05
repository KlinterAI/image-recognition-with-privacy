import ctypes
import os
import ioctl
import ioctl.linux

def make_ioctl_request():
    RNDGETENTCNT = ioctl.linux.IOR('R', 0x00, ctypes.c_int)
    rndgetentcnt = ioctl.ioctl_fn_ptr_r(RNDGETENTCNT, ctypes.c_int)

    fd = os.open('/dev/random', os.O_RDONLY)
    entropy_avail = rndgetentcnt(fd)
    print('entropy_avail:', entropy_avail)