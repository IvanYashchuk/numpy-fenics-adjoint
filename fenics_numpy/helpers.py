import fenics
import pyadjoint
import numpy as np

import warnings

from typing import Type, List, Union, Iterable, Callable, Tuple

FenicsVariable = Union[fenics.Constant, fenics.Function, pyadjoint.AdjFloat]


def fenics_to_numpy(fenics_var: FenicsVariable) -> np.array:
    """Convert FEniCS variable to numpy array.
    Serializes the input so that all MPI ranks have the same data."""
    if isinstance(fenics_var, fenics.Constant):
        return np.asarray(fenics_var.values())

    if isinstance(fenics_var, fenics.Function):
        fenics_vec = fenics_var.vector()
        if fenics_vec.mpi_comm().size > 1:
            data = fenics_vec.gather(np.arange(fenics_vec.size(), dtype="I"))
        else:
            data = fenics_vec.get_local()
        return np.asarray(data)

    if isinstance(fenics_var, fenics.GenericVector):
        if fenics_var.mpi_comm().size > 1:
            data = fenics_var.gather(np.arange(fenics_var.size(), dtype="I"))
        else:
            data = fenics_var.get_local()
        return np.asarray(data)

    if isinstance(fenics_var, (pyadjoint.AdjFloat, float)):
        return np.asarray(fenics_var)

    raise ValueError("Cannot convert " + str(type(fenics_var)))


def numpy_to_fenics(
    numpy_array: np.array, fenics_var_template: FenicsVariable
) -> FenicsVariable:  # noqa: C901
    """Convert numpy array to FEniCS variable.
    Distributes the input array across MPI ranks.
    Input:
        numpy_array (np.array): NumPy array to be converted to FEniCS type
        fenics_var_template (FenicsVariable): Templates for converting arrays to FEniCS type
    Output:
        fenucs_output (FenicsVariable): FEniCS representation of the input numpy_array
    """

    if isinstance(fenics_var_template, fenics.Constant):
        if numpy_array.shape == (1,):
            return type(fenics_var_template)(numpy_array[0])
        else:
            return type(fenics_var_template)(numpy_array)

    if isinstance(fenics_var_template, fenics.Function):
        function_space = fenics_var_template.function_space()

        u = type(fenics_var_template)(function_space)

        # assume that given numpy array is global array that needs to be distrubuted across processes
        # when FEniCS function is created
        fenics_size = u.vector().size()
        np_size = numpy_array.size

        if np_size != fenics_size:
            err_msg = (
                f"Cannot convert numpy array to Function:"
                f"Wrong size {numpy_array.size} vs {u.vector().size()}"
            )
            raise ValueError(err_msg)

        if numpy_array.dtype != np.float_:
            err_msg = (
                f"The numpy array must be of type {np.float_}, "
                "but got {numpy_array.dtype}"
            )
            raise ValueError(err_msg)

        range_begin, range_end = u.vector().local_range()
        numpy_array = np.asarray(numpy_array)
        local_array = numpy_array.reshape(fenics_size)[range_begin:range_end]
        u.vector().set_local(local_array)
        u.vector().apply("insert")
        return u

    if isinstance(fenics_var_template, pyadjoint.AdjFloat):
        return float(numpy_array)

    err_msg = f"Cannot convert numpy array to {fenics_var_template}"
    raise ValueError(err_msg)


def get_numpy_input_templates(
    fenics_input_templates: Iterable[FenicsVariable],
) -> List[np.array]:
    """Returns a list of numpy representations of the input templates"""
    numpy_input_templates = [fenics_to_numpy(x) for x in fenics_input_templates]
    return numpy_input_templates


def check_input(fenics_templates: FenicsVariable, *args: FenicsVariable) -> None:
    """Checks that the number of inputs arguments is correct"""
    n_args = len(args)
    expected_nargs = len(fenics_templates)
    if n_args != expected_nargs:
        raise ValueError(
            "Wrong number of arguments"
            " Expected {} got {}.".format(expected_nargs, n_args)
        )

    # Check that each input argument has correct dimensions
    numpy_templates = get_numpy_input_templates(fenics_templates)
    for i, (arg, template) in enumerate(zip(args, numpy_templates)):
        if arg.shape != template.shape:
            raise ValueError(
                "Expected input shape {} for input"
                " {} but got {}.".format(template.shape, i, arg.shape)
            )

    # Check that the inputs are of double precision
    for i, arg in enumerate(args):
        if arg.dtype != np.float64:
            raise TypeError(
                "All inputs must be type {},"
                " but got {} for input {}.".format(np.float64, arg.dtype, i)
            )


def convert_all_to_fenics(
    fenics_templates: Iterable[FenicsVariable], *args: np.array
) -> List[FenicsVariable]:
    """Converts input array to corresponding FEniCS variables"""
    fenics_inputs = []
    for inp, template in zip(args, fenics_templates):
        fenics_inputs.append(numpy_to_fenics(inp, template))
    return fenics_inputs
