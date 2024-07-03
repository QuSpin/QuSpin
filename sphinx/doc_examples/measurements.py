from __future__ import print_function, division

#
import sys, os

quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
#
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.tools.measurements import *
import numpy as np  # generic math functions

#
L = 12  # syste size
# coupling strenghts
J = 1.0  # spin-spin coupling
h = 0.8945  # x-field strength
g = 0.945  # z-field strength
# create site-coupling lists
J_zz = [[J, i, (i + 1) % L] for i in range(L)]  # PBC
x_field = [[h, i] for i in range(L)]
z_field = [[g, i] for i in range(L)]
# create static and dynamic lists
static_1 = [["x", x_field], ["z", z_field]]
static_2 = [["zz", J_zz], ["x", x_field], ["z", z_field]]
dynamic = []
# create spin-1/2 basis
basis = spin_basis_1d(L, kblock=0, pblock=1)
# set up Hamiltonian
H1 = hamiltonian(static_1, dynamic, basis=basis, dtype=np.float64)
H2 = hamiltonian(static_2, dynamic, basis=basis, dtype=np.float64)
# compute eigensystems of H1 and H2
E1, V1 = H1.eigh()
psi1 = V1[:, 14]  # pick any state as initial state
E2, V2 = H2.eigh()
# calculate entanglement entropy
Sent = ent_entropy(psi1, basis, chain_subsys=[1, 3, 6, 7, 11])
print(Sent["Sent_A"])
#
# calculate long-time (diagonal ensemble) expectations
Diag_Ens = diag_ensemble(L, psi1, E2, V2, Obs=H1, delta_t_Obs=True)
print(Diag_Ens["Obs_pure"], Diag_Ens["delta_t_Obs_pure"])
#
# time-evolve state by decomposing it in an eigensystem (E1,V1)
times = np.linspace(0.0, 5.0, 10)
psi1_time = ED_state_vs_time(psi1, E1, V1, times, iterate=False)
print(type(psi1_time))
# as above but using a generator
psi1_t = ED_state_vs_time(psi1, E1, V1, times, iterate=True)
print(type(psi1_t))
for i, psi1_n in enumerate(psi1_t):
    print("psi1_n is the state at time[%i]" % (i))
#
# calculate expectations of observables
Obs_time = obs_vs_time(psi1_time, times, dict(E1=H1, Energy2=H2))
print("Output keys are same as input keys:", Obs_time.keys())
E1_time = Obs_time["E1"]
#
# project Hamiltonian from `kblock=0` and `pblock=1` onto full Hilbert space
proj = basis.get_proj(np.float64)  # calculate projector
if sys.platform == "win32":
    H2_full = project_op(H2, proj, dtype=np.float64)["Proj_Obs"]
else:
    H2_full = project_op(H2, proj, dtype=np.float128)["Proj_Obs"]
print(
    "dimenions of symmetry-reduced and full Hilbert spaces are %i and %i "
    % (H2.Ns, H2_full.Ns)
)
#
# calculate mean level spacing of spectrum E2
d_2 = mean_level_spacing(E2)
print("mean level spacings are", d_2)
