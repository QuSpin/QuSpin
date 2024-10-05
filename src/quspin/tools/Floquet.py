# -*- coding: utf-8 -*-


# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp

import numpy as _np

from scipy.integrate import complex_ode
from joblib import delayed, Parallel
from numpy import vstack

import warnings

__all__ = ["Floquet_t_vec", "Floquet_t_vec"]

# warnings.warn("Floquet Package has not been fully tested yet, please report bugs to: https://github.com/weinbe58/qspin/issues.",UserWarning,stacklevel=3)


def _range_iter(start, stop, step):
    """'xrange' is replaced with 'range' in python 3. If python 2 is being used, range will cause memory overflow.
    This function is a work around to get the functionality of 'xrange' for both python 2 and 3 simultaineously.

    """
    from itertools import count

    counter = count(start, step)
    while True:
        i = next(counter)
        if i < stop:
            yield i
        else:
            break


def _evolve_cont(i, H, T, atol=1e-9, rtol=1e-9):
    """This function evolves the i-th local basis state under the Hamiltonian H up to period T.
    It is used to construct the stroboscpoic evolution operator.

    """

    psi0 = _np.zeros((H.Ns,), dtype=_np.complex128)
    psi0[i] = 1.0

    t_list = [0, T]
    nsteps = 1
    while nsteps < 1e7:
        try:
            psi_t = H.evolve(
                psi0, 0, t_list, eom="SE", iterate=False, atol=atol, rtol=rtol
            )
            return psi_t[:, -1]
        except:
            RuntimeError
        nsteps *= 10
        t_list = _np.linspace(0, T, num=nsteps + 1, endpoint=True)

    raise RuntimeError(
        "Ode solver takes more than {0:d} nsteps to complete time evolution. Cannot integrate ODE successfully.".format(
            nsteps
        )
    )

    """
	solver=complex_ode(H._hamiltonian__SO)
	solver.set_integrator('dop853', atol=atol,rtol=rtol,nsteps=nsteps) 
	solver.set_initial_value(psi0,t=0.0)
	t_list = [0,T]
	nsteps = 1
	while True:
		for t in t_list[1:]:
			solver.integrate(t)
			if solver.successful():
				if t == T:
					return solver.y
				continue
			else:
				break

		nsteps *= 10
		t_list = _np.linspace(0,T,num=nsteps+1,endpoint=True)

	"""


def _evolve_step_3(i, H_list, dt_list):
    """This function calculates the evolved state for Periodic Step (point 3. in def of 'evo_dict')."""

    psi0 = _np.zeros((H_list[0].Ns,), dtype=_np.complex128)
    psi0[i] = 1.0

    for dt, H in zip(dt_list, H_list):
        # can replace _sla.expm_multiply by tools.expm_multiply_parallel
        psi0 = _sla.expm_multiply(-1j * dt * H.tocsr(), psi0)

    return psi0


def _evolve_step_2(i, H, t_list, dt_list):
    """This function calculates the evolved state for Periodic Step (point 2. in def of 'evo_dict'."""

    psi0 = _np.zeros((H.Ns,), dtype=_np.complex128)
    psi0[i] = 1.0

    for t, dt in zip(t_list, dt_list):
        # can replace _sla.expm_multiply by tools.expm_multiply_parallel
        psi0 = _sla.expm_multiply(-1j * dt * H.tocsr(t), psi0)

    return psi0


### USING JOBLIB ###
def _get_U_cont(H, T, n_jobs, atol=1e-9, rtol=1e-9):

    sols = Parallel(n_jobs=n_jobs)(
        delayed(_evolve_cont)(i, H, T, atol, rtol) for i in _range_iter(0, H.Ns, 1)
    )
    return vstack(sols).T


def _get_U_step_3(H_list, dt_list, n_jobs):

    sols = Parallel(n_jobs=n_jobs)(
        delayed(_evolve_step_3)(i, H_list, dt_list)
        for i in _range_iter(0, H_list[0].Ns, 1)
    )

    return vstack(sols).T


def _get_U_step_2(H, t_list, dt_list, n_jobs):

    sols = Parallel(n_jobs=n_jobs)(
        delayed(_evolve_step_2)(i, H, t_list, dt_list) for i in _range_iter(0, H.Ns, 1)
    )

    return vstack(sols).T


class Floquet(object):
    """Calculates the Floquet spectrum, Floquet Hamiltonian and Floquet states.

    Loops over the basis states to compute the Floquet unitary :math:`U_F` (evolution operator over one period) for a
    periodically-driven system governed by the Hamiltonian :math:`H(t)=H(t+T)`:

    .. math::
            U_F=U(T,0)=\\mathcal{T}_t\\exp\\left(-i\\int_0^T\\mathrm{d}t H(t) \\right)

    with :math:`\\mathcal{T}_t\\exp` denoting the time-ordered exponential.

    Examples
    --------

    Consider the following periodically driven spin-1/2 Hamiltonian

    .. math::
            H(t) = \\left\\{
            \\begin{array}{cl} \\sum_j J\\sigma^z_{j+1}\\sigma^z_j + h\\sigma^z_j , &  t\\in[-T/4,T/4] \\newline
            \\sum_j g\\sigma^x_j, &  t \\in[T/4,3T/4]
            \\end{array}
            \\right\\}  \\mathrm{mod}\\ T

    where :math:`T=2\\pi/\\Omega` is the drive period. We choose the starting point of the evolution
    (or equivalently -- the driving phase) to be :math:`t=0`.

    The following snippet of code shows how to calculate the Floquet eigenstates and the corresponding quasienergies,
    using `evo_dict` variable, case ii (see below).

    .. literalinclude:: ../../doc_examples/Floquet_class-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(self, evo_dict, HF=False, UF=False, thetaF=False, VF=False, n_jobs=1, force_ONB=False):
        """Instantiates the `Floquet` class.

        Parameters
        ----------
        evo_dict : dict
                Dictionary which passes the different types of protocols to calculate the Floquet unitary.
                Depending on the protocol type, it contains the following keys:

                i) Periodic continuous protocol from a `hamiltonian` object.
                        * `H` : hamiltonian object to generate the time evolution.
                        * `T` : period of the protocol.
                        * `rtol` : (optional) relative tolerance for the ODE solver. (default = 1E-9)
                        * `atol` : (optional) absolute tolerance for the ODE solver. (default = 1E-9)

                ii) Periodic step protocol from a `hamiltonian` object.
                        * `H` : single hamiltonian object to generate the hamiltonians at each step. Periodic step drives can be encoded using a single function, e.g. :math:`\\mathrm{sign}(\\cos(\\Omega t))`.
                        * `t_list` : list of times to evaluate the hamiltonian at for each step.
                        * `dt_list` : list of time step durations for each step of the evolution.
                        * `T`: (optional) drive period used to compute the Floquet Hamiltonian `H_F`. If not specified, then `T=sum(dt_list)`. Use this option for periodic delta kicks.

                iii) Periodic step protocol from a list of hamiltonians.
                        * `H_list` : list of matrices to evolve with.
                        * `dt_list` : list of time step durations. Must be the same size as `H_list`.
                        * `T`: (optional) drive period used to compute the Floquet Hamiltonian `H_F`. If not specified, then `T=sum(dt_list)`. Use this option for periodic delta kicks.

        HF : bool
                Set to `True` to calculate and return Floquet Hamiltonian under attribute `_.HF`. Default is `False`.
        UF : bool
                Set to `True` to save evolution operator under attribute `_.UF`. Default is `False`.
        thetaF : bool
                Set to `True` to save eigenvalues of the evolution operator (Floquet phases) under attribute `_.thetaF`. Default is `False`.
        VF : bool
                Set to `True` to save Floquet states under attribute _.VF. Default is `False`.
        n_jobs : int, optional
                Sets the number of processors which are used when looping over the basis states to compute the Floquet unitary. Default is `False`.
        force_ONB : bool
                Set to `True` to run an extra QR decomposition to orthogonalize the Floquet states; only effective for `VF=True`. 
    
        """
        from quspin.operators import ishamiltonian

        variables = []
        if HF:
            variables.append("HF")
        if UF:
            variables.append("UF")
        if VF:
            variables.append("VF")
        if thetaF:
            variables.append("thetaF")

        if isinstance(evo_dict, dict):

            keys = evo_dict.keys()
            if (
                set(keys) == set(["H", "T"])
                or set(keys) == set(["H", "T", "atol"])
                or set(keys) == set(["H", "T", "rtol"])
                or set(keys) == set(["H", "T", "atol", "rtol"])
            ):

                H = evo_dict["H"]
                T = evo_dict["T"]
                self._atol = evo_dict.get("atol")
                self._rtol = evo_dict.get("rtol")

                if self._atol is None:
                    self._atol = 1e-12
                elif type(self._atol) is not float:
                    raise ValueError("expecting float for 'atol'.")

                if self._rtol is None:
                    self._rtol = 1e-12
                elif type(self._rtol) is not float:
                    raise ValueError("expecting float for 'rtol'.")

                if not ishamiltonian(H):
                    raise ValueError("expecting hamiltonian object for 'H'.")

                if not _np.isscalar(T):
                    raise ValueError("expecting scalar object for 'T'.")

                if _np.iscomplex(T):
                    raise ValueError("expecting real value for 'T'.")

                ### check if H is periodic with period T
                # define arbitrarily complicated weird-ass number

                t = _np.cos((_np.pi / _np.exp(0)) ** (1.0 / _np.euler_gamma))

                for func in H.dynamic:
                    if abs(func(t) - func(t + T)) > 1e5 * _np.finfo(_np.complex128).eps:
                        print(
                            abs(func(t) - func(t + T)),
                            1e3 * _np.finfo(_np.complex128).eps,
                        )
                        raise TypeError(
                            "Hamiltonian 'H' must be periodic with period 'T'!"
                        )

                if not (type(n_jobs) is int):
                    raise TypeError(
                        "expecting integer value for optional variable 'n_jobs'!"
                    )

                self._T = T

                # calculate evolution operator
                UF = _get_U_cont(H, self.T, n_jobs, atol=self._atol, rtol=self._rtol)

            elif set(keys) == set(["H", "t_list", "dt_list"]) or set(keys) == set(
                ["H", "t_list", "dt_list", "T"]
            ):
                H = evo_dict["H"]
                t_list = _np.asarray(evo_dict["t_list"], dtype=_np.float64)
                dt_list = _np.asarray(evo_dict["dt_list"], dtype=_np.float64)

                if t_list.ndim != 1:
                    raise ValueError("t_list must be 1d array.")

                if dt_list.ndim != 1:
                    raise ValueError("dt_list must be 1d array.")

                if "T" in set(keys):
                    self._T = evo_dict["T"]
                else:
                    self._T = dt_list.sum()

                if not ishamiltonian(H):
                    raise ValueError("expecting hamiltonian object for 'H'.")

                # calculate evolution operator
                UF = _get_U_step_2(H, t_list, dt_list, n_jobs)

            elif set(keys) == set(["H_list", "dt_list"]) or set(keys) == set(
                ["H_list", "dt_list", "T"]
            ):
                H_list = evo_dict["H_list"]
                dt_list = _np.asarray(evo_dict["dt_list"], dtype=_np.float64)

                if dt_list.ndim != 1:
                    raise ValueError("dt_list must be 1d array.")

                if "T" in set(keys):
                    self._T = evo_dict["T"]
                else:
                    self._T = dt_list.sum()

                if type(H_list) not in (list, tuple):
                    raise ValueError("expecting list/tuple for H_list.")

                if len(dt_list) != len(H_list):
                    raise ValueError(
                        "Expecting arguments 'H_list' and 'dt_list' to have the same length!"
                    )

                # calculate evolution operator
                UF = _get_U_step_3(H_list, dt_list, n_jobs)

            else:
                raise ValueError("evo_dict={0} is not correct format.".format(evo_dict))
        else:
            raise ValueError("evo_dict={0} is not correct format.".format(evo_dict))

        if "UF" in variables:
            self._UF = _np.copy(UF)

        if "HF" in variables:
            self._HF = 1j / self._T * _la.logm(UF)

        # find Floquet states and phases
        if "VF" in variables:
            thetaF, VF = _la.eig(UF, overwrite_a=True)
            # check and orthogonalise VF in degenerate subspaces
            if ( _np.any(_np.diff(_np.sort(thetaF)) < 1e3 * _np.finfo(thetaF.dtype).eps) ) or force_ONB:
                VF, _ = _la.qr(VF, overwrite_a=True)

            # https://math.stackexchange.com/questions/269164/diagonalizable-unitarily-schur-factorization
            # thetaF, VF = _la.schur(UF,overwrite_a=True,output='real')
            # thetaF=thetaF.diagonal()

            # calculate and order q'energies
            EF = _np.real(1j / self.T * _np.log(thetaF))
            # sort and order
            ind_EF = _np.argsort(EF)
            self._EF = _np.array(EF[ind_EF])
            self._VF = _np.array(VF[:, ind_EF])
            # clear up junk
            del VF
        else:
            thetaF = _la.eigvals(UF, overwrite_a=True)
            # calculate and order q'energies
            EF = _np.real(1j / self.T * _np.log(thetaF))
            ind_EF = _np.argsort(EF)
            self._EF = _np.array(EF[ind_EF])

        if "thetaF" in variables:
            # sort phases
            thetaF = _np.array(thetaF[ind_EF])
            self._thetaF = thetaF

    @property
    def T(self):
        """float: drive period."""
        return self._T

    @property
    def EF(self):
        """numpy.ndarray(float): ordered Floquet quasi-energies in interval :math:`[-\\Omega,\\Omega]`."""
        return self._EF

    @property
    def HF(self):
        """numpy.ndarray(float): Floquet Hamiltonian.

        Requires __init__ argument HF=True.

        """
        if hasattr(self, "_HF"):
            return self._HF
        else:
            raise AttributeError("missing atrribute 'HF'.")

    @property
    def UF(self):
        """numpy.ndarray(float): Floquet unitary.

        Requires __init__ argument UF=True.

        """
        if hasattr(self, "_UF"):
            return self._UF
        else:
            raise AttributeError("missing atrribute 'UF'.")

    @property
    def thetaF(self):
        """numpy.ndarray(float): Floquet eigenphases.

        Requires __init__ argument thetaF=True.

        """
        if hasattr(self, "_thetaF"):
            return self._thetaF
        else:
            raise AttributeError("missing atrribute 'thetaF'.")

    @property
    def VF(self):
        """numpy.ndarray(float): Floquet eigenbasis (in columns).

        Requires __init__ argument VF=True.

        """
        if hasattr(self, "_VF"):
            return self._VF
        else:
            raise AttributeError("missing atrribute 'VF'.")


class Floquet_t_vec(object):
    """Creates a Floquet time vector with fixed number of points per period.

    This time vector hits all stroboscopic times, and has many useful attributes. The time vector
    can be divided in three parts corresponding to three regimes of periodic evolution:
    ramp-up, constant and ramp-down.

    Particularly useful for studying periodically-driven systems.

    Examples
    --------

    The following code shows how to use the `Floquet_t_vec` class.

    .. literalinclude:: ../../doc_examples/Floquet_t_vec-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(self, Omega, N_const, len_T=100, N_up=0, N_down=0):
        """

        Parameters
        ----------
        Omega : float
                Drive frequency.
        N_const : int
                Number of time periods in the constant part (period) of the time vector.
        len_T : int
                Number of time points within a single period. N.B. the last period interval is assumed
                open on the right, i.e. [0,T) and the point T is NOT counted towards 'len_T'.
        N_up : int, optional
                Number of time periods in the up-part (period) of time vector.
        N_down : int, optional
                Number of time periods in the down-part (period) of time vector.

        """

        # total number of periods
        self._N = N_up + N_const + N_down
        # total length of a period
        self._len_T = len_T
        # driving period T
        self._T = 2.0 * _np.pi / Omega

        # define time vector
        n = _np.linspace(-N_up, N_const + N_down, self.N * len_T + 1)
        self._vals = self.T * n
        # total length of time vector
        self._len = self.vals.size
        # shape
        self._shape = self._vals.shape
        # time step
        self._dt = self.T / self.len_T
        # define index of period -N_up
        ind0 = 0  # int( _np.squeeze( (n==-N_up).nonzero() ) )

        # calculate stroboscopic times
        self._strobo = _strobo_times(self.vals, self.len_T, ind0)

        # define initial and final times and total duration
        self._i = self.vals[0]
        self._f = self.vals[-1]
        self._tot = self._f - self._i

        # if ramp is on, define more attributes
        if N_up > 0 and N_down > 0:
            t_up = self.vals[: self.strobo.inds[N_up]]
            self._up = _periodic_ramp(N_up, t_up, self.T, self.len_T, ind0)

            t_const = self.vals[
                self.strobo.inds[N_up] : self.strobo.inds[N_up + N_const] + 1
            ]
            ind0 = self.up.strobo.inds[-1] + self.len_T
            self._const = _periodic_ramp(N_const, t_const, self.T, self.len_T, ind0)

            t_down = self.vals[
                self.strobo.inds[N_up + N_const] + 1 : self.strobo.inds[-1] + 1
            ]
            ind0 = self.const.strobo.inds[-1] + self.len_T
            self._down = _periodic_ramp(N_down, t_down, self.T, self.len_T, ind0)

        elif N_up > 0:
            t_up = self.vals[: self.strobo.inds[N_up]]
            self._up = _periodic_ramp(N_up, t_up, self.T, self.len_T, ind0)

            t_const = self.vals[
                self.strobo.inds[N_up] : self.strobo.inds[N_up + N_const] + 1
            ]
            ind0 = self.up.strobo.inds[-1] + self.len_T
            self._const = _periodic_ramp(N_const, t_const, self.T, self.len_T, ind0)

        elif N_down > 0:
            t_const = self.vals[
                self.strobo.inds[N_up] : self.strobo.inds[N_up + N_const] + 1
            ]
            self._const = _periodic_ramp(N_const, t_const, self.T, self.len_T, ind0)

            t_down = self.vals[
                self.strobo.inds[N_up + N_const] + 1 : self.strobo.inds[-1] + 1
            ]
            ind0 = self.const.strobo.inds[-1] + self.len_T
            self._down = _periodic_ramp(N_down, t_down, self.T, self.len_T, ind0)

    def __iter__(self):
        return self.vals.__iter__()

    def __getitem__(self, s):
        return self._vals.__getitem__(s)

    def __str__(self):
        return str(self._vals)

    def __mul__(self, other):
        return self._vals * other

    def __div__(self, other):
        return self._vals / other

    def __truediv__(self, other):
        return self._vals / other

    def __len__(self):
        return self._vals.__len__()

    @property
    def N(self):
        """int: total number of periods."""
        return self._N

    @property
    def shape(self):
        """tuple: shape of array."""
        return self._shape

    @property
    def len_T(self):
        """int: number of time points within one period, assumed half-open; [0,T)."""
        return self._len_T

    @property
    def T(self):
        """float: drive period."""
        return self._T

    @property
    def vals(self):
        """np.ndarray(float): time vector values."""
        return self._vals

    @property
    def len(self):
        """int: length of time vector."""
        return self._len

    @property
    def dt(self):
        """float: time vector step size."""
        return self._dt

    @property
    def i(self):
        """float: initial time value."""
        return self._i

    @property
    def f(self):
        """foat: final time value."""
        return self._f

    @property
    def tot(self):
        """float: total time duration; `_.f - _.i` ."""
        return self._tot

    @property
    def strobo(self):
        """obj: calculates stroboscopic times in time vector with period length `len_T` and assigns them as
        attributes:

        _.strobo.inds : numpy.ndarray(int)
                indices of stroboscopic times (full periods).

        _.strobo.vals : numpy.ndarray(float)
                values of stroboscopic times (full periods).
        """
        return self._strobo

    @property
    def up(self):
        """obj: refers to time vector of up-part (regime).

        Inherits all attributes (e.g. `_.up.strobo.inds`) except `_.T`, `_.dt`, and `_.lenT`.

        Requires optional `__init___` parameter `N_up` to be specified.

        """
        if hasattr(self, "_up"):
            return self._up
        else:
            raise AttributeError("missing attribute 'up'")

    @property
    def const(self):
        """obj: refers to time vector of const-part (regime).

        Inherits all attributes (e.g. `_.const.strobo.inds`) except `_.T`, `_.dt`, and `_.lenT`.

        """
        if hasattr(self, "_const"):
            return self._const
        else:
            raise AttributeError("missing attribute 'const'")

    @property
    def down(self):
        """obj: refers to time vector of down-part (regime).

        Inherits all attributes (e.g. `_.down.strobo.inds`) except `_.T`, `_.dt`, and `_.lenT`.

        Requires optional __init___ parameter N_down to be specified.
        """
        if hasattr(self, "_down"):
            return self._down
        else:
            raise AttributeError("missing attribute 'down'")

    def get_coordinates(self, index):
        """Returns (period number, index within period) of the `Floquet_t_vec` value stored at `index`.

        Notes
        -----
                * This function finds the indegers (i,j), such that `t_evolve[t_evolve.strobo.inds[i-1] + j] = t_evolve[index]`.

                * The function may return wrong results if the spacing between two consecutive (i.e. nonstroboscopic) `Floquet_t_vec` values is smaller than `1E-15`.

        Parameters
        ----------
        index : int
                Index, to compute the `Floquet_t_vec` coordinates of.

        Returns
        -------
        tuple
                (i,j) such that `t_evolve[t_evolve.strobo.inds[i] + j] = t_evolve[index]`.

        Examples
        --------
        >>> t = Floquet_t_vec(10.0,10) # define a Floquet vector
        >>> index = 145 # pick a random index
        >>> print(t[index]) # check element
        >>> (i,j) = t.get_coordinates(index) # decompose index into stroboscopic coordinates
        >>> print( t[t.strobo.inds[i] + j] ) # we obtain back original element

        """

        t = self._vals[index]
        eps = 1e-15

        i = _np.searchsorted(self.strobo._vals, t + eps) - 1

        j = _np.where(
            _np.abs(t - i * self._T - self._vals[: self.strobo.inds[1]]) < eps
        )[0][0]

        return (i, j)


class _strobo_times:
    def __init__(self, t, len_T, ind0):
        """
        Calculates stroboscopic times in time vector t with period length len_T and assigns them as
        attributes.

        """
        # indices of strobo times
        self._inds = _np.arange(0, t.size, len_T).astype(int)
        # discrete stroboscopic t_vecs
        self._vals = t.take(self._inds)
        # update strobo indices to match shifted (ramped) ones
        self._inds += ind0

    @property
    def inds(self):
        return self._inds

    @property
    def vals(self):
        return self._vals

    def __iter__(self):
        return self.vals.__iter__()

    def __getitem__(self, s):
        return self._vals.__getitem__(s)

    def __str__(self):
        return str(self._vals)

    def __mul__(self, other):
        return self._vals * other

    def __div__(self, other):
        return self._vals / other

    def __truediv__(self, other):
        return self._vals / other

    def __len__(self):
        return self._vals.__len__()


class _periodic_ramp:
    def __init__(self, N, t, T, len_T, ind0):
        """Defines time vector attributes of each regime."""
        self._N = N  # total # periods
        self._vals = t  # time values
        self._i = self._vals[0]  # initial value
        self._f = self._vals[-1]  # final value
        self._tot = self._N * T  # total duration
        self._len = self._vals.size  # total length
        self._strobo = _strobo_times(self._vals, len_T, ind0)  # strobo attributes

    def __iter__(self):
        return self.vals.__iter__()

    def __getitem__(self, s):
        return self._vals.__getitem__(s)

    def __str__(self):
        return str(self._vals)

    def __mul__(self, other):
        return self._vals * other

    def __div__(self, other):
        return self._vals / other

    def __truediv__(self, other):
        return self._vals / other

    def __len__(self):
        return self._vals.__len__()

    @property
    def N(self):
        return self._N

    @property
    def vals(self):
        return self._vals

    @property
    def i(self):
        return self._i

    @property
    def f(self):
        return self._f

    @property
    def tot(self):
        return self._tot

    @property
    def len(self):
        return self._len

    @property
    def strobo(self):
        return self._strobo
