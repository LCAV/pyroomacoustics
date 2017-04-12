# @version: 1.0  date: 09/07/2015 by Robin Scheibler
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

'''
This file defines the main physical constants of the system
'''

# tolerance for computations
eps = 1e-10

# We implement the constants as a dictionnary so that they can
# be modified at runtime.
# The class Constants gives an interface to update the value of
# constants or add new ones.
_constants = {}
_constants_default = { 
    'c' : 343.0,      # speed of sound at 20 C in dry air
    'ffdist' : 10.,   # distance to the far field
    'fc_hp' : 300.,   # cut-off frequency of standard high-pass filter
    'frac_delay_length' : 81, # Length of the fractional delay filters used for RIR gen
    }


class Constants:
    '''
    A class to provide easy access package wide to user settable constants.
    
    Be careful of not using this in tight loops since it uses exceptions.
    '''

    def set(self, name, val):
        # add constant to dictionnary
        _constants[name] = val

    def get(self, name):

        try:
            v = _constants[name]
        except KeyError: 
            try:
                v = _constants_default[name]
            except KeyError:
                raise NameError(name + ': no such constant')

        return v


# the instanciation of the class
constants = Constants()

# Compute the speed of sound as a function 
# of temperature, humidity, and pressure
def calculate_speed_of_sound(t, h, p):
    '''
    Compute the speed of sound as a function of
    temperature, humidity and pressure

    Parameters
    ----------
    t: 
        temperature [Celsius]
    h: 
        relative humidity [%]
    p: 
        atmospheric pressure [kpa]

    Returns
    -------

    Speed of sound in [m/s]
    '''

    # using crude approximation for now
    return 331.4 + 0.6*t + 0.0124*h

