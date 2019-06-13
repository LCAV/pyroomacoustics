## TODO

* Set default time threshold in ray tracing to -1 so that it is only based on energy
* Add back t0 in the simulation

## Test wether the ray tracing follows Sabine

TODO

## Track which parts of simulation are finished

TODO

## Create a new API for constructing rooms

* ShoeBox (similar to current one, but with ray tracing)
* from floorplan (equivalent to from_corners + extrude)
* from walls (same as current Room constructor)
* from file, this would definitely the easiest way to handle more complicated
  non-shoebox models. A challenge is to have a file format that handles materials.
  There is currently one example for the STL file format.


## Methods that let you construct a room and their API

Attributes:

    # old
    fs
    max_order
    sources
    mics

    # deprecated ?
    t0
    sigma2_awgn

    # new
    ray_trace_args
    air_absorption

    # internals
    - simulation state (ism, rt, rir, sim)
    - keep track of change in sources/microphones
      to know which rir to recompute
    - mono-band vs multi-band

    # environment
    - Is interface with temperature/humidity the best ?

    - It is also useful to be able to set the speed of sound directly
      for example to recreate some special conditions.

    - One possible choice is to have temp/hum as the default in the
      constructor, and support directly set the speed of sound via
      a member function interface


### Room

Constructs directly from a bunch of walls

* old interface:
  
        walls,
        fs=8000,
        t0=0.,
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None

* new interface:
    
        self,
        walls,
        fs=8000,
        temperature=25.,
        humidity=70.,
        c=None,
        air_absorption=None,
        max_order=1,
        ray_trace_args=None,
        sources=None,
        mics=None,

* compatible new interface

        self,
        walls,
        fs=8000,
        t0=0.,
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None,
        temperature=None,
        humidity=None,
        air_absorption=None,
        ray_trace_args=None,

### from_corners

Constructs a 2D room (floorplan) from a bunch of points forming a polygon

* old interface:
  
        corners,
        absorption=0.,
        fs=8000,
        t0=0.,
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None

* new interface:
    
        corners,
        absorption=None,
        materials=None,
        fs=8000,
        **kwargs,

### extrude

Normally follows a construction by `from_corners` to make a 3D room from a 2D floor plan.

* old interface:
  
        height,
        v_vec=None,
        absorption=0.

* new interface:
  
        self,
        height,
        v_vec=None,
        materials=None,
        absorption=None,

### Shoebox

Constructor of the Shoebox class

* old interface:
        p,
        fs=8000,
        t0=0.,
        absorption=0.,
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None

* new interface:
  
        p,
        absorption=None,  # deprecated
        materials=None,
        fs=8000,
        temperature=25.,
        humidity=70.,
        c=None,
        air_absorption=None,
        max_order=1,
        ray_trace_args=None,
        sources=None,
        mics=None,
