from quspin.basis import (
    spin_basis_general,
    boson_basis_general,
    spinless_fermion_basis_general,
    spinful_fermion_basis_general,
)
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

    R = np.rot90(s.reshape(Lx, Ly), axes=(0, 1)).reshape(N_2d)  # rotate

    P_x = x + Lx * (Ly - y - 1)  # reflection about x-axis
    P_y = (Lx - x - 1) + Lx * y  # reflection about y-axis

    Z = -(s + 1)  # spin inversion

    #
    ###### setting up bases ######

    basis_boson = boson_basis_general(
        N_2d,
        make_basis=False,
        Nb=N_2d // 4,
        sps=2,
        kxblock=(T_x, 0),
        kyblock=(T_y, 0),
        rblock=(R, 0),
        pxblock=(P_x, 0),
        pyblock=(P_y, 0),
    )

    basis_boson_full = boson_basis_general(
        N_2d,
        make_basis=True,
        Nb=N_2d // 4,
        sps=2,
    )


    basis_spin = spin_basis_general(
        N_2d,
        pauli=False,
        make_basis=False,
        Nup=N_2d // 2,
        kxblock=(T_x, 0),
        kyblock=(T_y, 0),
        rblock=(R, 0),
        pxblock=(P_x, 0),
        pyblock=(P_y, 0),
        zblock=(Z, 0),
    )

    basis_spin_full = spin_basis_general(
        N_2d,
        pauli=False,
        make_basis=True,
        Nup=N_2d // 2,
    )

    basis_fermion = spinless_fermion_basis_general(
        N_2d,
        make_basis=False,
        Nf=N_2d // 2,
        kxblock=(T_x, 0),
        kyblock=(T_y, 0),
        rblock=(R, 0),
        pxblock=(P_x, 0),
        pyblock=(P_y, 0),
    )

    basis_fermion_full = spinless_fermion_basis_general(
        N_2d,
        make_basis=True,
        Nf=N_2d // 2,
    )


    basis_spinful_fermion = spinful_fermion_basis_general(
        N_2d,
        make_basis=False,
        Nf=(N_2d // 8, N_2d // 8),
        kxblock=(T_x, 0),
        kyblock=(T_y, 0),
        rblock=(R, 0),
        pxblock=(P_x, 0),
        pyblock=(P_y, 0),
    )

    basis_spinful_fermion_full = spinful_fermion_basis_general(
        N_2d,
        make_basis=True,
        Nf=(N_2d // 8, N_2d // 8),
    )


    bases_2d = [basis_boson, basis_spin, basis_fermion, basis_spinful_fermion]
    bases_2d_full = [
        basis_boson_full,
        basis_spin_full,
        basis_fermion_full,
        basis_spinful_fermion_full,
    ]


    for i, (basis_2d, basis_2d_full) in enumerate(zip(bases_2d, bases_2d_full)):
        # grab states of full basis
        states = basis_2d_full.states

        # check function
        ref_states = basis_2d.representative(states)
        ref_states = np.sort(np.unique(ref_states))[::-1]

        norms = basis_2d.normalization(ref_states)
        mask = np.abs(norms) != 0.0

        # check inplace function
        ref_states_inplace = np.zeros_like(states)
        basis_2d.representative(states, out=ref_states_inplace)
        ref_states_inplace = np.sort(np.unique(ref_states_inplace))[::-1]

        out_dtype = np.min_scalar_type(np.iinfo(basis_2d.dtype).max * basis_2d._pers.prod())
        norms_inplace = np.zeros_like(ref_states_inplace, dtype=out_dtype)
        basis_2d.normalization(ref_states_inplace, out=norms_inplace)
        mask_inplace = np.abs(norms_inplace) != 0.0

        # make full basis to compare to
        basis_2d.make(Ns_block_est=20000)

        np.testing.assert_allclose(
            basis_2d.states - ref_states[mask],
            0.0,
            atol=1e-5,
            err_msg="failed representative test!",
        )
        np.testing.assert_allclose(
            basis_2d.states - ref_states_inplace[mask_inplace],
            0.0,
            atol=1e-5,
            err_msg="failed inplace representative test!",
        )

        # check g_out and sign_out flags
        ref_states, g_out, sign_out = basis_2d.representative(
            states, return_g=True, return_sign=True
        )
        ref_states, g_out = basis_2d.representative(states, return_g=True)
        ref_states, sign_out = basis_2d.representative(states, return_sign=True)

        r_out = np.zeros_like(states)
        basis_2d.representative(
            states, out=r_out, return_g=True, return_sign=True
        )
        basis_2d.representative(states, out=r_out, return_g=True)
        basis_2d.representative(states, out=r_out, return_sign=True)


        print("test {} passed".format(i))
