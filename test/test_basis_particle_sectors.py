from quspin.basis import (
    spin_basis_1d,
    boson_basis_1d,
    spinless_fermion_basis_1d,
    spinful_fermion_basis_1d,
)
from quspin.basis import (
    spin_basis_general,
    boson_basis_general,
    spinless_fermion_basis_general,
    spinful_fermion_basis_general,
)
from itertools import product
import numpy as np

def test():

    for L in [6, 7]:

        # symmetry-free

        spin_basis_1d(L=L, Nup=range(0, L, 2))
        spin_basis_general(N=L, Nup=range(0, L, 2))

        boson_basis_1d(L=L, Nb=range(0, L, 2))
        boson_basis_general(N=L, Nb=range(0, L, 2))

        spinless_fermion_basis_1d(L=L, Nf=range(0, L, 2))
        spinless_fermion_basis_general(N=L, Nf=range(0, L, 2))

        spinful_fermion_basis_1d(L=L, Nf=product(range(0, L, 2), range(0, L, 2)))
        spinful_fermion_basis_general(
            N=L, Nf=product(range(0, L, 2), range(0, L, 2))
        )

        # symmetry-ful

        t = (np.arange(L) + 1) % L

        spin_basis_1d(L=L, Nup=range(0, L, 2), kblock=0)
        spin_basis_general(N=L, Nup=range(0, L, 2), kblock=(t, 0))

        boson_basis_1d(L=L, Nb=range(0, L, 2), kblock=0)
        boson_basis_general(N=L, Nb=range(0, L, 2), kblock=(t, 0))

        spinless_fermion_basis_1d(L=L, Nf=range(0, L, 2), kblock=0)
        spinless_fermion_basis_general(N=L, Nf=range(0, L, 2), kblock=(t, 0))

        spinful_fermion_basis_1d(
            L=L, Nf=product(range(0, L, 2), range(0, L, 2)), kblock=0
        )
        spinful_fermion_basis_general(
            N=L, Nf=product(range(0, L, 2), range(0, L, 2)), kblock=(t, 0)
        )

        print("passed particle number sectors test")


if __name__ == "__main__":
    test()