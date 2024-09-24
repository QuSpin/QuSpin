from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import boson_basis_1d
from quspin.tools.measurements import ent_entropy
import numpy as np  # generic math functions

def test():
    ##### define model parameters #####
    L = 6  # system size

    J = 1.0  # hopping
    U = np.sqrt(2)  # interactions strenth


    # define site-coupling lists
    interaction = [[U / 2.0, i, i] for i in range(L)]  # PBC
    chem_pot = [[-U / 2.0, i] for i in range(L)]  # PBC
    hopping = [[J, i, (i + 1) % L] for i in range(L)]  # PBC

    #### define hcb model
    basis = boson_basis_1d(L=L, Nb=L, sps=L + 1, kblock=0, pblock=1)

    # Hubbard-related model
    static = [["+-", hopping], ["-+", hopping], ["n", chem_pot], ["nn", interaction]]

    H = hamiltonian(static, [], basis=basis, dtype=np.float32)
    E, V = H.eigh()


    ent_entropy({"V_states": V}, basis, chain_subsys=range(L // 2))["Sent"]
    # print(Sent)
