# Yuxuan Zhang, University of Florida
from collections import namedtuple
# Prototype of named tuple that describes a LED
LED = namedtuple("LED", ["Name", "bandwidth", "halfDelta"])
# The band order in the captured image follows the
# same order as listed below
LED_LIST = [
	LED(name="U-Violet", bandwidth=395, halfDelta=10),
	LED(name="Blue"    , bandwidth=466, halfDelta=15),
	LED(name="Green"   , bandwidth=520, halfDelta=15),
	LED(name="Y-Green" , bandwidth=573, halfDelta=20),
	LED(name="Yellow"  , bandwidth=585, halfDelta=20),
	LED(name="Orange"  , bandwidth=600, halfDelta=20),
	LED(name="Red"     , bandwidth=660, halfDelta=17),
	LED(name="Infrared", bandwidth=940, halfDelta=40),
]
# This is the band numbers of the reference camera's output
REF_BANDS = list(range(402, 998 + 2, 2))
