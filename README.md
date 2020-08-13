# numpy-fenics-adjoint &middot; [![Build](https://github.com/ivanyashchuk/numpy-fenics-adjoint/workflows/CI/badge.svg)](https://github.com/ivanyashchuk/numpy-fenics-adjoint/actions?query=workflow%3ACI+branch%3Amaster) [![Coverage Status](https://coveralls.io/repos/github/IvanYashchuk/numpy-fenics-adjoint/badge.svg?branch=master)](https://coveralls.io/github/IvanYashchuk/numpy-fenics-adjoint?branch=master)
Easy interoperability with Automatic Differentiation libraries through NumPy interface to FEniCS.

## Overview
This package provides a high-level interface for evaluating derivatives of [FEniCS](http://fenicsproject.org) models.
It is intended to be used as the backend for extending Automatic Differentiation libraries to support FEniCS solvers.

Automatic tangent linear and adjoint solvers for FEniCS problems are derived with [dolfin-adjoint/pyadjoint](http://www.dolfin-adjoint.org/en/latest/).
These solvers make it possible to use forward and reverse modes Automatic Differentiation with FEniCS.

This package is used for building bridges between FEniCS and JAX in [jax-fenics-adjoint](https://github.com/IvanYashchuk/jax-fenics-adjoint/). Stay tuned for the PyMC3 (Theano), Julia's ChainRule.jl, PyTorch integrations.

Current limitations:
* Composition of forward and reverse modes for higher-order derivatives are not implemented yet.
* Differentiation wrt Dirichlet boundary conditions and mesh coordinates is not implemented yet.

## Example
Here is the demonstration of solving the [Poisson's PDE](https://en.wikipedia.org/wiki/Poisson%27s_equation)
on the 2D square domain and calculating the result of multiplying a vector with the solution Jacobian matrix (_du/df_) using the reverse (adjoint) mode Automatic Differentiation.
```python
import numpy as np

import fenics
import fenics_adjoint
import ufl

from functools import partial

from fenics_numpy import evaluate_primal, evaluate_vjp
from fenics_numpy import fenics_to_numpy, numpy_to_fenics

# Create mesh for the unit square domain
n = 10
mesh = fenics_adjoint.UnitSquareMesh(n, n)

# Define discrete function spaces and functions
V = fenics.FunctionSpace(mesh, "CG", 1)
W = fenics.FunctionSpace(mesh, "DG", 0)

# Define FEniCS template representation of NumPy input
templates = (fenics_adjoint.Function(W),)

# This function takes FEniCS types as arguments and returns a FEniCS Function (solution)
def fenics_solve(f):
    # This function inside should be traceable by fenics_adjoint
    u = fenics_adjoint.Function(V, name="PDE Solution")
    v = fenics.TestFunction(V)
    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    F = (inner(grad(u), grad(v)) - f * v) * dx
    bcs = [fenics_adjoint.DirichletBC(V, 0.0, "on_boundary")]
    fenics_adjoint.solve(F == 0, u, bcs)
    return u

# Let's build a decorator which transforms NumPy input to FEniCS types input
# and returns NumPy representation of FEniCS output
numpy_fenics_solve = partial(evaluate_primal, fenics_solve, templates)

# Let's create a vector of ones with size equal to the number of cells in the mesh
f = np.ones(W.dim())
u = numpy_fenics_solve(f)[0] # u is a NumPy array now
u_fenics = numpy_to_fenics(u, fenics.Function(V)) # we need to explicitly provide template function for conversion

# Now let's evaluate the vector-Jacobian product
numpy_output, fenics_output, fenics_inputs, tape = numpy_fenics_solve(f)
g = np.ones_like(numpy_output)

# `vjp_out` is the result of (implicitly) multiplying the vector `g` with the solution Jacobian du/df
vjp_out = evaluate_vjp(g, fenics_output, fenics_inputs, tape)
```

Check the `tests/` folder for the additional usage examples.

## Installation
First install [FEniCS](http://fenicsproject.org).
Then install [dolfin-adjoint](http://www.dolfin-adjoint.org/en/latest/) with:

    python -m pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@master

Then install [numpy-fenics-adjoint](https://github.com/IvanYashchuk/numpy-fenics-adjoint) with:

    python -m pip install git+https://github.com/IvanYashchuk/numpy-fenics-adjoint@master


## Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/IvanYashchuk/numpy-fenics-adjoint/issues/new

## Contributing

Pull requests are welcome from everyone.

Fork, then clone the repository:

    git clone https://github.com/IvanYashchuk/numpy-fenics-adjoint.git

Make your change. Add tests for your change. Make the tests pass:

    pytest tests/

Check the formatting with `black` and `flake8`. Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/IvanYashchuk/numpy-fenics-adjoint/pulls
