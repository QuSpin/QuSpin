from quspin.basis import (
    spin_basis_general,
    boson_basis_general,
    spinless_fermion_basis_general,
    spinful_fermion_basis_general,
)
import numpy as np

from quspin.operators._make_hamiltonian import _consolidate_static

def test():
    #
    ###### define model parameters ######
    J1 = 1.0  # spin=spin interaction
    J2 = 0.5  # magnetic field strength
    Lx, Ly = 4, 2  # linear dimension of 2d lattice
    N_2d = Lx * Ly  # number of sites
    #
    ###### setting up user-defined symmetry transformations for 2d lattice ######
    s = np.arange(N_2d)  # sites [0,1,2,....]
    x = s % Lx  # x positions for sites
    y = s // Lx  # y positions for sites

    T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
    T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction

    T_a = (x + 1) % Lx + Lx * ((y + 1) % Ly)  # translation along anti-diagonal
    T_d = (x - 1) % Lx + Lx * ((y + 1) % Ly)  # translation along diagonal

    P_x = x + Lx * (Ly - y - 1)  # reflection about x-axis
    P_y = (Lx - x - 1) + Lx * y  # reflection about y-axis

    Z = -(s + 1)  # spin inversion

    #####


    # setting up site-coupling lists
    J1_list = [[J1, i, T_x[i]] for i in range(N_2d)] + [
        [J1, i, T_y[i]] for i in range(N_2d)
    ]
    J2_list = [[J2, i, T_d[i]] for i in range(N_2d)] + [
        [J2, i, T_a[i]] for i in range(N_2d)
    ]
    #
    static = [
        ["++", J1_list],
        ["--", J1_list],
        ["zz", J1_list],
        ["++", J2_list],
        ["--", J2_list],
        ["zz", J2_list],
    ]

    static_spfs = [
        ["++|", J1_list],
        ["--|", J1_list],
        ["|++", J1_list],
        ["|--", J1_list],
        ["z|z", J1_list],
        ["++|", J2_list],
        ["--|", J2_list],
        ["|++", J2_list],
        ["|--", J2_list],
        ["z|z", J2_list],
    ]

    static_list = _consolidate_static(static)
    static_list_spfs = _consolidate_static(static_spfs)


    def compare(static_list, basis, basis_op):
        for opstr, indx, J in static_list:
            ME, bra, ket = basis.Op_bra_ket(opstr, indx, J, np.float64, basis_op.states)
            ME_op, row, col = basis_op.Op(opstr, indx, J, np.float64)
            np.testing.assert_allclose(
                bra - basis_op[row],
                0.0,
                atol=1e-5,
                err_msg="failed bra/row in Op_bra_cket test!",
            )
            np.testing.assert_allclose(
                ket - basis_op[col],
                0.0,
                atol=1e-5,
                err_msg="failed ket/col in Op_bra_ket test!",
            )
            np.testing.assert_allclose(
                ME - ME_op, 0.0, atol=1e-5, err_msg="failed ME in Op_bra_ket test!"
            )


    for Np in [None, 2, N_2d - 1, [N_2d // 4, N_2d // 8]]:

        basis = spin_basis_general(
            N_2d,
            make_basis=False,
            Nup=Np,
            kxblock=(T_x, 0),
            kyblock=(T_y, 0),
            pxblock=(P_x, 0),
            pyblock=(P_y, 0),
            zblock=(Z, 0),
        )

        basis_op = spin_basis_general(
            N_2d,
            make_basis=True,
            Nup=Np,
            kxblock=(T_x, 0),
            kyblock=(T_y, 0),
            pxblock=(P_x, 0),
            pyblock=(P_y, 0),
            zblock=(Z, 0),
        )

        compare(static_list, basis, basis_op)
        print("passed spins")

        basis = spinless_fermion_basis_general(
            N_2d,
            make_basis=False,
            Nf=Np,
            kxblock=(T_x, 0),
            kyblock=(T_y, 0),
            pxblock=(P_x, 0),
            pyblock=(P_y, 0),
        )

        basis_op = spinless_fermion_basis_general(
            N_2d,
            make_basis=True,
            Nf=Np,
            kxblock=(T_x, 0),
            kyblock=(T_y, 0),
            pxblock=(P_x, 0),
            pyblock=(P_y, 0),
        )

        compare(static_list, basis, basis_op)
        print("passed spinless fermios")

        basis = boson_basis_general(
            N_2d,
            make_basis=False,
            Nb=Np,
            sps=3,
            kxblock=(T_x, 0),
            kyblock=(T_y, 0),
            pxblock=(P_x, 0),
            pyblock=(P_y, 0),
        )

        basis_op = boson_basis_general(
            N_2d,
            make_basis=True,
            Nb=Np,
            sps=3,
            kxblock=(T_x, 0),
            kyblock=(T_y, 0),
            pxblock=(P_x, 0),
            pyblock=(P_y, 0),
        )

        compare(static_list, basis, basis_op)
        print("passed bosons")

        if Np is None:
            Nf = Np
        elif type(Np) is list:
            Nf = list(zip(Np, Np))
        else:
            Nf = (Np, Np)

        basis = spinful_fermion_basis_general(
            N_2d,
            make_basis=False,
            Nf=Nf,
            kxblock=(T_x, 0),
            kyblock=(T_y, 0),
            pxblock=(P_x, 0),
            pyblock=(P_y, 0),
        )

        basis_op = spinful_fermion_basis_general(
            N_2d,
            make_basis=True,
            Nf=Nf,
            kxblock=(T_x, 0),
            kyblock=(T_y, 0),
            pxblock=(P_x, 0),
            pyblock=(P_y, 0),
        )

        compare(static_list_spfs, basis, basis_op)
        print("passed spinful fermios")
