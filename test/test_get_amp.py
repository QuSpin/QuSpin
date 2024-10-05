from quspin.operators import hamiltonian
from quspin.basis import (
    spin_basis_general,
    boson_basis_general,
    spinless_fermion_basis_general,
    spinful_fermion_basis_general,
)
from quspin.basis.transformations import square_lattice_trans
import numpy as np

def test():
    #
    ###### define model parameters ######
    Lx, Ly = 4, 4  # linear dimension of 2d lattice
    N_2d = Lx * Ly  # number of sites
    #
    ###### setting up user-defined symmetry transformations for 2d lattice ######
    s = np.arange(N_2d)  # sites [0,1,2,....]
    x = s % Lx  # x positions for sites
    y = s // Lx  # y positions for sites

    T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
    T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction
    Z = -(s + 1)  # spin inversion

    J = np.sqrt(7)
    J_p = [[+J, i, T_x[i]] for i in range(N_2d)] + [[+J, i, T_y[i]] for i in range(N_2d)]
    J_n = [[-J, i, T_x[i]] for i in range(N_2d)] + [[-J, i, T_y[i]] for i in range(N_2d)]

    lattice_trans = square_lattice_trans(Lx, Ly)
    allowed_sectors = lattice_trans.allowed_blocks_iter()

    for ii, basis_dict in enumerate(allowed_sectors):

        ###### setting up bases ######

        basis_boson = boson_basis_general(
            N_2d, make_basis=False, Nb=N_2d // 4, sps=2, **basis_dict
        )

        basis_boson_full = boson_basis_general(
            N_2d,
            make_basis=False,
            Nb=N_2d // 4,
            sps=2,
        )

        basis_spin = spin_basis_general(
            N_2d, pauli=False, make_basis=False, Nup=N_2d // 2, zblock=(Z, 0), **basis_dict
        )

        basis_spin_full = spin_basis_general(
            N_2d,
            pauli=False,
            make_basis=False,
            Nup=N_2d // 2,
        )

        basis_fermion = spinless_fermion_basis_general(
            N_2d, make_basis=False, Nf=N_2d // 2, **basis_dict
        )

        basis_fermion_full = spinless_fermion_basis_general(
            N_2d,
            make_basis=False,
            Nf=N_2d // 2,
        )

        basis_spinful_fermion = spinful_fermion_basis_general(
            N_2d, make_basis=False, Nf=(N_2d // 8, N_2d // 8), **basis_dict
        )

        basis_spinful_fermion_full = spinful_fermion_basis_general(
            N_2d,
            make_basis=False,
            Nf=(N_2d // 8, N_2d // 8),
        )

        bases_2d = [basis_boson, basis_spin, basis_fermion, basis_spinful_fermion]
        bases_2d_full = [
            basis_boson_full,
            basis_spin_full,
            basis_fermion_full,
            basis_spinful_fermion_full,
        ]

        for i, (basis_2d, basis_2d_full, basis_2d_made, basis_2d_full_made) in enumerate(
            zip(bases_2d, bases_2d_full, bases_2d, bases_2d_full)
        ):

            basis_2d_made.make(Ns_block_est=16000)
            basis_2d_full_made.make(Ns_block_est=16000)

            if i in [2, 6]:  # spinless fermions
                static = [["zz", J_p], ["+-", J_p], ["-+", J_n]]
            elif i in [3, 7]:  # spinful fermions
                static = [
                    ["zz|", J_p],
                    ["|zz", J_p],
                    ["+-|", J_p],
                    ["-+|", J_n],
                    ["|+-", J_p],
                    ["|-+", J_n],
                ]
            else:  # boson, spin
                static = [["zz", J_p], ["+-", J_p], ["-+", J_p]]

            print(
                "# of states",
                i,
                basis_2d_made.Ns,
                basis_2d_full_made.Ns,
                basis_2d_made.__class__,
                basis_2d_made.blocks,
            )

            H = hamiltonian(static, [], basis=basis_2d_made, dtype=np.complex128)
            E_GS, V_GS = H.eigsh(k=1, which="SA", maxiter=10000)  # ,sigma=1E-6

            states = basis_2d_made.states.copy()
            inds = [np.where(basis_2d_full.states == r)[0][0] for r in states]
            psi_GS = V_GS[:, 0].copy()
            out=basis_2d.get_amp(states,amps=psi_GS,mode='representative') # updates psi_GS in place!!!
            psi_tmp = basis_2d_made.get_vec(V_GS[:, 0], pcon=True, sparse=False)[inds]
            np.testing.assert_allclose(
                psi_GS - psi_tmp,
                0.0,
                atol=1e-14,
                err_msg="failed representative mode comparison!",
            )

            states_full = basis_2d_full_made.states.copy()
            psi_GS_full = basis_2d_made.get_vec(
                V_GS[:, 0], pcon=True, sparse=False
            )  # V_GS_full[:,0].astype(np.complex128)
            basis_2d.get_amp(states_full, amps=psi_GS_full, mode="full_basis")
            rep_states = basis_2d_made.representative(basis_2d_full_made.states)
            psi_GS_full = psi_GS_full[inds]
            np.testing.assert_allclose(
                psi_GS_full - V_GS[:, 0],
                0.0,
                atol=1e-14,
                err_msg="failed full_basis mode comparison!",
            )
            np.testing.assert_allclose(
                rep_states - states_full,
                0.0,
                atol=1e-14,
                err_msg="failed full_basis mode states comparison!",
            )

            print("test {0}/{1} passed".format(ii, i))


#test()

