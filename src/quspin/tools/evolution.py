# -*- coding: utf-8 -*-


# need linear algebra packages
import numpy as _np
from functools import partial as _partial
from scipy.integrate import ode
from numpy.linalg import norm

# needed for isinstance only
from quspin.tools.expm_multiply_parallel_core import ExpmMultiplyParallel

# define alias for backward compatibility
expm_multiply_parallel = ExpmMultiplyParallel

__all__ = ["ED_state_vs_time", "evolve", "ExpmMultiplyParallel", "expm_multiply_parallel"]

##### below are the routines for arbitary user-defimed time evolution.


def ED_state_vs_time(psi, E, V, times, iterate=False):
    """Calculates the time evolution of initial state using a complete eigenbasis.

    The time evolution is carried out under the Hamiltonian :math:`H` with eigenenergies `E` and eigenstates `V`.

    Examples
    --------

    The following example shows how to time-evolve a state :math:`\\psi` using the entire eigensystem
    :math:`(E_1,V_1)` of a Hamiltonian :math:`H_1=\\sum_j hS^x_j + g S^z_j`.

    .. literalinclude:: ../../doc_examples/ED_state_vs_time-example.py
            :linenos:
            :language: python
            :lines: 7-

    Parameters
    ----------
    psi : numpy.ndarray
            Initial state.
    V : numpy.ndarray
            Unitary matrix containing all eigenstates of the Hamiltonian :math:`H` in its columns.
    E : numpy.ndarray
            Eigenvalues of the Hamiltonian :math:`H`, listed in the order which corresponds to the columns of `V`.
    times : numpy.ndarray
            Vector of time to evaluate the time evolved state at.
    iterate : bool, optional
            If set to `True`, the function returns the generator of the time evolved state.

    Returns
    -------
    obj
            Either of the following:
                    * `numpy.ndarray` with the time evolved states as rows.
                    * `generator` which generates time-dependent states one by one.

    """
    psi = _np.squeeze(_np.asarray(psi))

    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError("'V' must be a square matrix")

    if V.shape[0] != len(E):
        raise TypeError(
            "Number of eigenstates in 'V' must equal number of eigenvalues in 'E'!"
        )
    if psi.shape[0] != len(E):
        raise TypeError("Variables 'psi' and 'E' must have the same dimension!")

    if psi.ndim == 2:
        if psi.shape[0] != psi.shape[1]:
            raise ValueError("mixed states must be square!")

    if psi.ndim > 2:
        raise ValueError("psi must be 1 or 2 dimension array.")

    if _np.isscalar(times):
        TypeError("Variable 'times' must be a array or iter like object!")

    times = -1j * _np.asarray(times)

    # define generator of time-evolved state in basis V2
    def pure_t_iter(V, psi, times):
        # a_n: probability amplitudes
        # times: time vector
        a_n = V.T.conj().dot(psi)
        for t in times:
            yield V.dot(_np.exp(E * t) * a_n)

    def mixed_t_iter(V, psi, times):
        # a_n: probability amplitudes
        # times: time vector
        rho_d = V.T.conj().dot(psi.dot(V))
        for t in times:
            exp_t = _np.exp(t * E)
            yield _np.einsum(
                "ij,j,jk,k,lk->il", V, exp_t, rho_d, exp_t.conj(), V.conj()
            )

    if psi.ndim == 1:
        if iterate:
            return pure_t_iter(V, psi, times)
        else:
            c_n = V.T.conj().dot(psi)

            Ntime = len(times)
            Ns = len(E)

            # generate [[-1j*times[0], ..., -1j*times[0]], ..., [-1j*times[-1], ..., -1j*times[01]]
            psi_t = _np.broadcast_to(times, (Ns, Ntime)).T
            # [[-1j*E[0]*times[0], ..., -1j*E[-1]*times[0]], ..., [-1j*E[0]*times[-1], ..., -1j*E[-1]*times[-1]]
            psi_t = psi_t * E
            # [[exp(-1j*E[0]*times[0]), ..., exp(-1j*E[-1]*times[0])], ..., [exp(-1j*E[0]*times[-1]), ..., exp(-1j*E[01]*times[01])]
            _np.exp(psi_t, psi_t)

            # [[c_n[0]exp(-1j*E[0]*times[0]), ..., c_n[-1]*exp(-1j*E[-1]*times[0])], ..., [c_n[0]*exp(-1j*E[0]*times[-1]), ...,c_n[o]*exp(-1j*E[01]*times[01])]
            psi_t *= c_n

            # for each vector trasform back to original basis
            psi_t = V.dot(psi_t.T)

            return psi_t  # [ psi(times[0]), ...,psi(times[-1]) ]
    else:
        if iterate:
            return mixed_t_iter(V, psi, times)
        else:
            Ntime = len(times)
            Ns = len(E)

            rho_d = V.T.conj().dot(psi.dot(V))

            # generate [[-1j*times[0], ..., -1j*times[0]], ..., [-1j*times[-1], ..., -1j*times[01]]
            exp_t = _np.broadcast_to(times, (Ns, Ntime)).T
            # [[-1j*E[0]*times[0], ..., -1j*E[-1]*times[0]], ..., [-1j*E[0]*times[-1], ..., -1j*E[-1]*times[-1]]
            exp_t = exp_t * E
            # [[exp(-1j*E[0]*times[0]), ..., exp(-1j*E[-1]*times[0])], ..., [exp(-1j*E[0]*times[-1]), ..., exp(-1j*E[01]*times[01])]
            _np.exp(exp_t, exp_t)

            return _np.einsum(
                "ij,tj,jk,tk,lk->ilt", V, exp_t, rho_d, exp_t.conj(), V.conj()
            )


def evolve(
    v0,
    t0,
    times,
    f,
    solver_name="dop853",
    real=False,
    stack_state=False,
    verbose=False,
    imag_time=False,
    iterate=False,
    f_params=(),
    **solver_args,
):
    """Implements (imaginary) time evolution for a user-defined first-order ODE.

    The function can be used to study nonlinear semiclassical dynamics. It can also serve as a pre-configured
    ODE solver in python, without any relation to other QuSpin objects.

    Examples
    --------

    The following example shows how to use the `evolve()` function to solve the periodically-driven
    Gross-Pitaevskii equation (GPE) on a one-imensional lattice. The GPE has a linear part, comprising the
    kinetic energy and the external potentials (e.g. a harmonic trap), and a nonlinear part which describes
    the interactions.

    Below, in a few steps we show how to use the functionality of the `evolve()` function to solve the GPE
    on a one-dimensional lattice for periodically-driven particles in a harmonic trap:

    .. math::
            i\\dot\\varphi_j(t) = -J\\left( e^{-iA\\sin\\Omega t}\\varphi_{j-1}(t) + e^{+iA\\sin\\Omega t}\\varphi_{j+1}(t) \\right) + \\kappa_\\mathrm{trap}\\varphi_j(t) + U|\\varphi_j(t)|^2\\varphi_j(t)

    where :math:`j` labels the lattice sites, :math:`J` is the lattice hopping amplitude, :math:`\\kappa_\\mathrm{trap}` is
    the strength of the harmonic trap, and :math:`U` -- the mean-field interaction.

    Let us start by defining the single-particle Hamiltonian :math:`H(t)`.


    .. literalinclude:: ../../doc_examples/evolve-example.py
            :linenos:
            :language: python
            :lines: 7-47

    Next, we define the GPE

    .. math::
            -i\\dot\\varphi(t) = H(t)\\varphi(t) + U |\\varphi(t)|^2 \\varphi(t)

    and solve it using `evolve()`:

    .. literalinclude:: ../../doc_examples/evolve-example.py
            :linenos:
            :language: python
            :lines: 49-61
            :lineno-start: 43

    Since the GPE is a complex-valued equation, the above code requires the use of a complex-valued ODE solver
    [which is done by `evolve()` under the hood, so long as no solver is explicitly specified].

    An alternative way
    to solve the GPE using a real-valued solver might be useful to speed-up the computation. This can be achieved
    by decomposing the condensate wave function into a real and imaginary part, and proceeds as follows:

    The goal is to solve:

    .. math::
            -i\\dot\\varphi(t) = H(t)\\varphi(t) + U |\\varphi(t)|^2 \\varphi(t)

    for the complex-valued :math:`\\varphi(t)` by re-writing it as a real-valued vector `phi=[u,v]` where
    :math:`\\varphi(t) = u(t) + iv(t)`. The real and imaginary parts, :math:`u(t)` and :math:`v(t)`, have the same array dimension as
    :math:`\\phi(t)`.

    In the most general form, the single-particle Hamiltonian can be decomposed as
    :math:`H(t)= H_{stat} + f(t)H_{dyn}`, with a complex-valued driving function :math:`f(t)`. Then, the GPE can be cast in
    the following real-valued form:

    .. math::
            \\dot u(t) = +\\left[H_{stat} + U\\left(|u(t)|^2 + |v(t)|^2\\right) \\right]v(t) + Re[f(t)]H_{dyn}v(t) + Im[f(t)]H_{dyn}u(t)
    .. math::
            \\dot v(t) = -\\left[H_{stat} + U\\left(|u(t)|^2 + |v(t)|^2\\right) \\right]u(t) - Re[f(t)]H_{dyn}u(t) + Im[f(t)]H_{dyn}v(t)


    .. literalinclude:: ../../doc_examples/evolve-example.py
            :linenos:
            :language: python
            :lines: 63-
            :lineno-start: 59

    The flag `stack_state=True` is required for `evolve()` to handle the complex-valued initial condition properly,
    as well as to put together the output solution as a complex-valued vector in the end. Since the real-valued ODE solver
    allows to parse ODE parameters, we can include them in the user-defined ODE function and use the
    flag `f_params`. Notice the elegant way python allows one to circumvent the usage of this variable in the
    complex-valued example above.

    Parameters
    ----------
    v0 : numpy.ndarray
            Initial state.
    t0 : float
            Initial time.
    times : numpy.ndarray
            Vector of times to compute the time-evolved state at.
    f : :obj:`function`
            User-defined function to solve first-order ODE (see Examples):

            .. math::
                    v'(t) = f(v(t),t)\\qquad v(t_0)=v_0
    f_params : tuple, optional
            A tuple to pass all parameters of the function `f` to ODE solver. Default is `f_params=()`.
    iterate : bool, optional
            If set to `True`, creates a generator object for the time-evolved the state. Default is `False`.
    solver_name : str, optional
            Scipy solver integrator name. Default is `dop853`.

            See `scipy integrator (solver) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_ for other options.
    solver_args : dict, optional
            Dictionary with additional `scipy integrator (solver) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_ arguments.
    real : bool, optional
            Flag to determine if `f` is real or complex-valued. Default is `False`.
    imag_time : bool, optional
            Must be set to `True` when `f` defines imaginary-time evolution, in order to normalise the state
            at each time in `times`. Default is `False`.
    stack_state : bool, optional
            If `f` is written to take care of real and imaginary parts separately (see Examples), this flag
            will return a single complex-valued solution instead of the real and imaginary parts separately.
            Default is `False`.
    verbose : bool, optional
            If set to `True`, prints normalisation of state at teach time in `times`.

    Returns
    -------
    obj
            Can be either one of the following:
                    * numpy.ndarray containing evolved state against time.
                    * generator object for time-evolved state (requires `iterate = True`).

    """

    ndim = v0.ndim
    if ndim > 2:
        raise ValueError("state mush have ndim < 3.")

    shape0 = v0.shape

    if ndim == 2:
        v0 = v0.ravel()
        shape0_ravelled = v0.shape

    if _np.iscomplexobj(times):
        raise ValueError("times must be real number(s).")

    n = _np.linalg.norm(
        v0
    )  # needed for imaginary time to preserve the proper norm of the state.

    if stack_state:
        if imag_time:
            raise ValueError("imag_time is not compatible with stack_state.")

        complex_valued = False
        v1 = v0.copy()
        if ndim == 1:
            v0 = _np.zeros(2 * shape0[0], dtype=v1.real.dtype)
            v0[: shape0[0]] = v1.real
            v0[shape0[0] :] = v1.imag
        else:
            v0 = _np.zeros(2 * shape0_ravelled[0], dtype=v1.real.dtype)
            v0[: shape0_ravelled[0]] = v1.real
            v0[shape0_ravelled[0] :] = v1.imag

        solver = ode(f)  # y_f = f(t,y,*args)
        solver.set_f_params(*f_params)
    elif real:
        complex_valued = False
        solver = ode(f)  # y_f = f(t,y,*args)
        solver.set_f_params(*f_params)
    else:
        complex_valued = True
        # check if array is contiguous (required by memory view)
        try:
            v0 = v0.astype(_np.complex128, copy=False).view(_np.float64)
        except ValueError:
            # copy initial state v0 to make it contiguous
            v0 = v0.astype(_np.complex128, copy=True).view(_np.float64)
        solver = ode(_cmplx_f)  # y_f = f(t,y,*args)
        solver.set_f_params(f, f_params)

    if solver_name in ["dop853", "dopri5"]:
        if solver_args.get("nsteps") is None:
            solver_args["nsteps"] = _np.iinfo(_np.int32).max
        if solver_args.get("rtol") is None:
            solver_args["rtol"] = 1e-9
        if solver_args.get("atol") is None:
            solver_args["atol"] = 1e-9

    solver.set_integrator(solver_name, **solver_args)
    solver.set_initial_value(v0, t0)

    output_args = (complex_valued, stack_state, imag_time, n, shape0)

    if _np.isscalar(times):
        return _evolve_scalar(solver, v0, t0, times, *output_args)
    else:
        if iterate:
            return _evolve_iter(solver, v0, t0, times, verbose, *output_args)
        else:
            return _evolve_list(solver, v0, t0, times, verbose, *output_args)


def _cmplx_f(t, y, f, f_params):
    yc = y.view(_np.complex128)
    return f(t, yc, *f_params).view(_np.float64)


def _format_output(y, complex_valued, stack_state, imag_time, n, shape0):
    Ns = shape0[0]
    if stack_state:
        yout = y[:Ns].astype(_np.complex128).reshape(shape0)
        yout[...] += 1j * y[Ns:].reshape(shape0)
    elif complex_valued:
        yout = y.view(_np.complex128).reshape(shape0)
    else:
        yout = y.reshape(shape0)

    if imag_time:
        yout /= norm(yout, axis=0) / n

    return yout


def _evolve_scalar(solver, v0, t0, time, *output_args):
    if time == t0:
        return _format_output(v0, *output_args)

    solver.integrate(time)
    if solver.successful():
        return _format_output(solver._y, *output_args)
    else:
        raise RuntimeError(
            "failed to evolve to time {0}, nsteps might be too small".format(time)
        )


def _evolve_list(solver, v0, t0, times, verbose, *output_args):
    shape0 = output_args[-1]
    Ns = shape0[0]

    if output_args[0] or output_args[1]:
        v = _np.empty((len(times),) + shape0, dtype=_np.complex128, order="C")
    else:
        v = _np.empty((len(times),) + shape0, dtype=_np.float64, order="C")

    for i, t in enumerate(times):

        if t == t0:
            y_fmt = _format_output(v0, *output_args)
            if verbose:
                print(
                    "evolved to time {0}, norm of state(s) {1}".format(
                        t, norm(y_fmt, axis=0)
                    )
                )
            v[i, ...] = y_fmt
            continue

        solver.integrate(t)
        if solver.successful():
            y_fmt = _format_output(solver._y, *output_args)
            if verbose:
                print(
                    "evolved to time {0}, norm of state(s) {1}".format(
                        t, norm(y_fmt, axis=0)
                    )
                )
            v[i, ...] = y_fmt
        else:
            raise RuntimeError(
                "failed to evolve to time {0}, nsteps might be too small".format(t)
            )

    if v.ndim == 2:
        v = v.transpose()
    else:
        v = v.transpose((1, 2, 0))

    return v


def _evolve_iter(solver, v0, t0, times, verbose, *output_args):
    shape0 = output_args[-1]
    Ns = shape0[0]

    for i, t in enumerate(times):
        if t == t0:
            y_fmt = _format_output(v0, *output_args)
            if verbose:
                print(
                    "evolved to time {0}, norm of state(s) {1}".format(
                        t, norm(y_fmt, axis=0)
                    )
                )
            yield y_fmt
            continue

        solver.integrate(t)
        if solver.successful():
            y_fmt = _format_output(solver._y, *output_args)
            if verbose:
                print(
                    "evolved to time {0}, norm of state(s) {1}".format(
                        t, norm(y_fmt, axis=0)
                    )
                )
            yield y_fmt
        else:
            raise RuntimeError(
                "failed to evolve to time {0}, nsteps might be too small".format(t)
            )
