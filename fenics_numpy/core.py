import fenics
import fenics_adjoint
import pyadjoint

import numpy as np

import functools

from .helpers import (
    numpy_to_fenics,
    fenics_to_numpy,
    get_numpy_input_templates,
    check_input,
    convert_all_to_fenics,
)
from .helpers import FenicsVariable

from typing import Type, List, Union, Iterable, Callable, Tuple


def evaluate_primal(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    *args: np.array,
) -> Tuple[np.array, FenicsVariable, Tuple[FenicsVariable], pyadjoint.Tape]:
    """Computes the output of a fenics_function and saves a corresponding gradient tape
    Input:
        fenics_function (callable): FEniCS function to be executed during the forward pass
        fenics_templates (iterable of FenicsVariable): Templates for converting arrays to FEniCS types
        args (tuple): NumPy array representation of the input to fenics_function
    Output:
        numpy_output (np.array): NumPy array representation of the output from fenics_function(*fenics_inputs)
        fenics_output (AdjFloat or Function): FEniCS representation of the output from fenics_function(*fenics_inputs)
        fenics_inputs (list of FenicsVariable): FEniCS representation of the input args
        tape (pyadjoint.Tape): pyadjoint's saved computational graph
    """

    check_input(fenics_templates, *args)
    fenics_inputs = convert_all_to_fenics(fenics_templates, *args)

    # Create tape associated with this forward pass
    tape = pyadjoint.Tape()
    pyadjoint.set_working_tape(tape)
    fenics_output = fenics_function(*fenics_inputs)

    if isinstance(fenics_output, tuple):
        raise ValueError("Only single output from FEniCS function is supported.")

    numpy_output = np.asarray(fenics_to_numpy(fenics_output))
    return numpy_output, fenics_output, fenics_inputs, tape


# Below unicode symbols are used to distinguish between input and ouput sensitivities
# See http://www.juliadiff.org/ChainRules.jl/dev/FAQ.html
# Δx is the input to a propagator, (i.e a seed for a pullback; or a perturbation for a pushforward)
# ∂x is the output of a propagator
# dx could be either
# Here dx is used for the output of a propagator since ∂x is not a valid name for python variables


def evaluate_vjp(
    dnumpy_output: np.array,
    fenics_output: FenicsVariable,
    fenics_inputs: Iterable[FenicsVariable],
    tape: pyadjoint.Tape,
) -> Tuple[np.array]:
    """Computes the gradients of the output with respect to the inputs.
    Input:
        Δfenics_output (np.array): NumPy array representation of the tangent covector to multiply transposed jacobian with
        fenics_output (AdjFloat or Function): FEniCS representation of the output from fenics_function(*fenics_inputs)
        fenics_inputs (list of FenicsVariable): FEniCS representation of the input args
        tape (pyadjoint.Tape): pyadjoint's saved computational graph
    Output:
        dnumpy_inputs (list of np.array):
            NumPy array representation of the `Δfenics_output` times jacobian
            of fenics_function(*fenics_inputs) wrt to every fenics_input
    """
    # Convert tangent covector (adjoint variable) to a FEniCS variable
    Δfenics_output = numpy_to_fenics(dnumpy_output, fenics_output)
    if isinstance(Δfenics_output, (fenics.Function, fenics_adjoint.Function)):
        Δfenics_output = Δfenics_output.vector()

    tape.reset_variables()
    fenics_output.block_variable.adj_value = Δfenics_output
    with tape.marked_nodes(fenics_inputs):
        tape.evaluate_adj(markings=True)
    dfenics_inputs = (fi.block_variable.adj_value for fi in fenics_inputs)

    # Convert FEniCS gradients to NumPy array representation
    dnumpy_inputs = tuple(
        None if dfi is None else np.asarray(fenics_to_numpy(dfi))
        for dfi in dfenics_inputs
    )

    return dnumpy_inputs


def evaluate_jvp(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    numpy_inputs: Iterable[np.array],
    Δnumpy_inputs: Iterable[np.array],
) -> Tuple[np.array]:
    """Computes the primal FEniCS function together with the corresponding tangent linear model.
    Note that Δnumpy_inputs are sometimes referred to as tangent vectors.
    """

    numpy_output, fenics_output, fenics_inputs, tape = evaluate_primal(
        fenics_function, fenics_templates, *numpy_inputs
    )

    # Now tangent (pushforward) evaluation!
    tape.reset_variables()

    Δfenics_inputs = convert_all_to_fenics(fenics_inputs, *Δnumpy_inputs)
    for fi, Δfi in zip(fenics_inputs, Δfenics_inputs):
        fi.block_variable.tlm_value = Δfi

    tape.evaluate_tlm()

    dfenics_output = fenics_output.block_variable.tlm_value
    dnumpy_output = fenics_to_numpy(dfenics_output)

    return numpy_output, dnumpy_output
