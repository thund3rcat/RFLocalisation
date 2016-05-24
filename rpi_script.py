"""This module is used on a raspberry pi to read
PSD values on a certein frequency
"""

import rf

EAR_PI = rf.RFear(435e6)

EAR_PI.rpi_get_power(1, 128)


