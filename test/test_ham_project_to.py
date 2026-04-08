r"""
Test the hamiltonian.project_to() method with static and dynamic Hamiltonians
using different basis projections and data types for both spin and bosonic systems.
"""

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d, boson_basis_1d
import numpy as np


def drive_func(t, omega):
    return np.sin(omega * t)

def test_hamiltonian_spin_project_to():
    r"""
    Test Spin system with fixed number of spins and projection onto k=0 (momentum) block and p=1 (parity) block
    """
    dtypes = {
        "float64": np.float64,
        "complex128": np.complex128,
    }

    atols = {"float64": 1e-13, "complex128": 1e-13}
    L = 5

    for dtype_name, dtype in dtypes.items():
        atol = atols[dtype_name]

        # Create full basis and momentum-blocked basis
        basis_full = spin_basis_1d(L=L)
        basis_k0_p0 = spin_basis_1d(L=L, kblock=0, pblock=1, a=1)

        # Define operators
        J = 1  # spin  interaction strength
        h = 0.9  # longitudinal field strength
        g = 0.8  # transverse field strength

        J_zz = [[J, i, (i + 1) % L] for i in range(L)]
        z_field = [[h, i] for i in range(L)]
        x_field = [[g, i] for i in range(L)]

        static = [["z", z_field], ["zz", J_zz]]

        dynamic = [["x", x_field, drive_func, [1.0]]]  # omega = 1.0

        # Create Hamiltonians
        H_full_static = hamiltonian(
            static_list=static, dynamic_list=[], basis=basis_full, dtype=dtype
        )
        H_full = hamiltonian(
            static_list=static, dynamic_list=dynamic, basis=basis_full, dtype=dtype
        )
        H_k0_direct = hamiltonian(
            static_list=static, dynamic_list=dynamic, basis=basis_k0_p0, dtype=dtype
        )

        # Test projection using projector matrix
        P_k0 = basis_k0_p0.get_proj(
            dtype=dtype,
        )

        H_k0_projected_static = H_full_static.project_to(P_k0)
        H_k0_projected = H_full.project_to(P_k0)

        # Compare at t=0
        np.testing.assert_allclose(
            H_k0_projected_static.toarray(time=0),
            H_k0_projected.toarray(time=0),
            atol=atol,
            err_msg=f"Failed spin hamiltonian projection comparison for static part for {dtype_name}!",
        )

        # Compare at different times
        for t in [0.0, 0.5, 1.0 * np.pi]:
            np.testing.assert_allclose(
                H_k0_direct.toarray(time=t),
                H_k0_projected.toarray(time=t),
                atol=atol,
                err_msg=f"Failed spin hamiltonian projection comparison for dynamic part projection comparison at t={t:.4f} for {dtype_name}!",
            )

        # Test eigenvalues
        E_direct = H_k0_direct.eigvalsh()
        E_projected = H_k0_projected.eigvalsh()

        np.testing.assert_allclose(
            E_direct,
            E_projected,
            atol=atol,
            err_msg=f"Failed projected spin hamiltonian eigenvalue comparison for {dtype_name}!",
        )


def test_hamiltonian_boson_project_to():
    r""" 
    Test Bosonic system with fixed particle number and projection onto k=0 block
    """

    dtypes = {
        "float64": np.float64,
        "complex128": np.complex128,
    }

    atols = {"float64": 1e-13, "complex128": 1e-13}
    L_boson = 3
    Nb = 2
    sps = 4

    for dtype_name, dtype in dtypes.items():
        atol = atols[dtype_name]

        # Create full and restricted bases
        basis_full = boson_basis_1d(L=L_boson, Nb=Nb, sps=sps)
        basis_k0 = boson_basis_1d(L=L_boson, Nb=Nb, sps=sps, kblock=0, a=1)

        # Bose-Hubbard model with drive
        U = 1.0
        Kamp = 0.5
        omega = 1.0

        hopping = [[-1.0, i, (i + 1) % L_boson] for i in range(L_boson)]
        interaction1 = [[U / 2, i, i] for i in range(L_boson)]
        interaction2 = [[-U / 2, i] for i in range(L_boson)]
        drive_terms = [[Kamp, i] for i in range(L_boson)]

        static = [
            ["+-", hopping],
            ["-+", hopping],
            ["nn", interaction1],
            ["n", interaction2],
        ]
        dynamic = [["n", drive_terms, drive_func, [omega]]]

        # Create Hamiltonians
        H_full_static = hamiltonian(static, [], basis=basis_full, dtype=dtype)
        H_full = hamiltonian(static, dynamic, basis=basis_full, dtype=dtype)
        H_k0_direct = hamiltonian(static, dynamic, basis=basis_k0, dtype=dtype)

        # Test projection
        P_proj_k0 = basis_k0.get_proj(dtype=dtype, pcon=True)

        H_k0_projected = H_full.project_to(P_proj_k0)
        H_k0_projected_static = H_full_static.project_to(P_proj_k0)

        # Compare at t=0 - projected full hamiltonian static part vs dynamic part (t=0 -> same as static)
        np.testing.assert_allclose(
            H_k0_projected_static.toarray(time=0),
            H_k0_projected.toarray(time=0),
            atol=atol,
            err_msg=f"Failed bosonic hamiltonian projection comparison for static part for {dtype_name}!",
        )

        # Compare at different times - directly projected and full hamiltonian projected
        for t in [0.0, 0.5, 1.0 * np.pi]:
            np.testing.assert_allclose(
                H_k0_direct.toarray(time=t),
                H_k0_projected.toarray(time=t),
                atol=atol,
                err_msg=f"Failed bosonic hamiltonian projection comparison for dynamic part projection comparison at t={t:.4f} for {dtype_name}!",
            )

        # Compare eigenvalues
        E_direct = H_k0_direct.eigvalsh()
        E_projected = H_k0_projected.eigvalsh()

        np.testing.assert_allclose(
            E_direct,
            E_projected,
            atol=atol,
            err_msg=f"Failed projected bosonic hamiltonian eigenvalue comparison for  {dtype_name}!",
        )


if __name__ == "__main__":
    test_hamiltonian_spin_project_to()
    test_hamiltonian_boson_project_to()
    print("All 1D spin and bosonic Hamiltonian project_to tests passed!")
