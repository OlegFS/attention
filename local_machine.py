"""
=========================================
Module for determining the local machine.
=========================================
"""

from os import popen


KNOWN_MACs = {'00:21:9b:03:37:ee': 'macgyver',
              '00:1d:09:bb:11:46': 'rabbit-mcrabbit',
              '00:18:f3:ef:2a:fa': 'tosca',
              '00:30:48:b8:53:86': 'sesto',
              '00:10:18:d9:3c:74': 'publio',
              '00:25:90:61:6d:42': 'pamina',
              '00:25:90:10:7f:16': 'despina',
              '00:30:48:de:9b:7a': 'anna',
              '00:30:48:b8:52:3e': 'vitellia', 
              '00:22:15:01:d0:0c': 'elvira',
              '00:25:90:c1:b8:ba': 'romeo',
              '00:25:90:21:79:20': 'fidelio'}


class MachineIdentityError(IOError):
    """Exception Class for unknown computer"""
    pass

def get_mac(interface='eth0'):
    """Gets the mac address of the interface (default=eth0)."""
    # get ifconfig
    pipe = popen('/sbin/ifconfig')
    data = pipe.read()
    pipe.close()
    # parse to get the MAC address
    mac = data.split(interface)[1].split('HWaddr ')[1].split(' ')[0]
    return mac.lower()

def get_local_machine():
    """Get the name of the local machine.

    Raises MachineIdentityError if there is no name.

    """
    mac = get_mac()
    try:
        machine_name = KNOWN_MACs[mac]
    except KeyError:
        raise MachineIdentityError('unknown local machine')
        #machine_name = 'unknown'
    return machine_name

def is_machine(name):
    """Checks if the machine name is `name`"""
    machine_name = get_local_machine()
    return machine_name == name

def is_lcv():
    """Checks if the machine is on the lcv network"""
    machine_name = get_local_machine()
    return machine_name in ['publio', 'pamina', 'fidelio', 'sesto', 'vitellia',
            'tosca', 'annio', 'elvira', 'despina', 'anna', 'despina', 'romeo']
