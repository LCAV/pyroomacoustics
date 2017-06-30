
def calculate_speed_of_sound(t, h, p): 
    ''' 
    Compute the speed of sound as a function of
    temperature, humidity and pressure

    Arguments
    ---------

    t: temperature [Celsius]
    h: relative humidity [%]
    p: atmospheric pressure [kpa]

    Return
    ------

    Speed of sound in [m/s]
    '''

    # using crude approximation for now
    return 331.4 + 0.6*t + 0.0124*h
