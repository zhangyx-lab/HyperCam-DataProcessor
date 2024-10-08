# Yuxuan Zhang, University of Florida
from collections import namedtuple
# Color profiles for our camera
COLORS = [
    "Ultra_Violet", # 0
    "Blue",			# 1
    "Green",		# 2
    "Yellow_Green", # 3
    "Yellow",		# 4
    "Orange",		# 5
    "Red",			# 6
    "Infrared"		# 7
]
# Prototype of named tuple that describes a LED
LED = namedtuple("LED", ["name", "bandwidth", "delta"])
# The band order in the captured image follows the
# same order as listed below
LED_LIST = [
	LED(name="U-Violet", bandwidth=395, delta=20),
	LED(name="Blue"    , bandwidth=466, delta=30),
	LED(name="Green"   , bandwidth=520, delta=30),
	LED(name="Y-Green" , bandwidth=573, delta=40),
	LED(name="Yellow"  , bandwidth=585, delta=40),
	LED(name="Orange"  , bandwidth=600, delta=40),
	LED(name="Red"     , bandwidth=660, delta=34),
	LED(name="Infrared", bandwidth=940, delta=80),
]
# This is the band numbers of the reference camera's output
REF_BANDS = list(range(402, 998 + 2, 2))
