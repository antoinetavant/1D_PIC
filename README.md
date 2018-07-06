# 1D Particle-In-Cells Electrostatic simulation
### For plasma simulation.

A PIC simulation code moves particles (electrons and ions) on a given domain. It couple the motion of the particles with the generated electric and magnetic field.


#### This Simulation code is 1D Electrostatic :
* Only the Poisson equation is solved, generating a self-consistent electric field
* The particles are moved in one direction, even is there velocity is defined in 3D

#### Applications
We can use it to study interesting 1D kinetic effects such as plasma instabilities, to improve existing models, or to illustrate plasma lectures.

**The advantage** of a PIC code is that the equations are simple to solve.
**The disadvantage** of a PIC code is that it as to deal with **LOTS** of particles. The performance is a big issue

## Getting Started

### Prerequisites
This code only uses Python and standard Python modules:
> Numpy,
> Numba,
> tkinter,
> matplotlib,
> Pytest

Because performance is a really big issue, I compared different functions from different modules, but most of the time the fastest solution was to use `Numba.jit`.
Is is true by example for interpolation and density estimation.

### Installing

You just need to clone the repo, and you should be able to execute both main files `main.py` and `Main.ipynb`


## Contributing

Feel free to fork and contribute to the code.
However, I should specify that this project's main objective is to test solutions for the [PlasmaPy project](https://github.com/PlasmaPy/PlasmaPy). Hence, contributing to PlasmaPy could be more interesting.

### Todo list

* Physic and Solvers
- [x] Electrostatic particle pusher (Leap-frog scheme)
- [ ] Electromagnetic particle pusher (Boris scheme)
- [x] Poisson solver (Thomas's algorithm)
- [x] Wall boundary conditions
- [x] periodic boundary conditions

* interface and user experience
- [ ] GUI for initialisation
- [x] GUI of the evolution of the simulation
- [x] diagnostics (data output)
- [ ] Restars

* Software part
- [x] Tests
- [ ] 95 % Coverage
- [ ] Performance profiling
- [ ] Continuous integration
- [ ] Parallelistation

## Authors
* [Antoine Tavant](https://github.com/antoinelpp)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

I have to think about it ! The list of my references is long, even though this project is only my own...
I'd say mostly Vivien Croes and Trevor Lafleur, for now.

I'll improve the list later.
