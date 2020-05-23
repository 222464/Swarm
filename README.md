# Swarm

A recurrent convolutional neural network trained with multi-armed bandit swarm intelligence (BSN, bandit swarm networks, as described [here](https://twistedkeyboardsoftware.com/?p=147)).

Python bindings (recommended) available [here](https://github.com/222464/PySwarm)

## Installation

### CMake

Version 3.1+ of [CMake](https://cmake.org/) is required when building the library.

### Building

> cd Swarm  
> mkdir build  
> cd build  
> cmake ..  
> make  
> make install

The `BUILD_SHARED_LIBS` boolean cmake option can be used to create dynamic/shared object library (default is to create a _static_ library). On Linux it's recommended to add `-DBUILD_SHARED_LIBS=ON`

`make install` can be run to install the library. `make uninstall` can be used to uninstall the library.

On **Windows** systems it is recommended to use `cmake-gui` to define which generator to use and specify optional build parameters, such as `CMAKE_INSTALL_PREFIX`.

## License

MIT license. See [LICENSE.md](./LICENSE.md) for more information.

This library depends on OpenMP for multithreading.