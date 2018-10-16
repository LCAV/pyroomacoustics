import pyroomacoustics as pra
import sys

# verify if the compiled libroom is available or not
if pra.c_package.libroom_available:
    print('Compiled libroom available')
    sys.exit(0)
else:
    print('Compiled libroom unavailable')
    sys.exit(1)
