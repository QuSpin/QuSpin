"""Regression test: int_to_state(..., bracket_notation=False) for sps > 10.

When sps > 10 a single site can hold an occupation >= 10. The non-bracket
Fock string must remain a base-sps digit string (digits 0-9 then a-z) so that
it round-trips through state_to_int()/index() (which parse it via
int(s, sps)).

Before the fix, int_to_state built a space-padded decimal string and then
stripped the spaces, so e.g. |10 0 0 0> collapsed to the ambiguous "10000"
(indistinguishable from a length-5 state) and did NOT round-trip:
state_to_int("10000") raised "s must be representive state in basis".
"""

import numpy as np
from quspin.basis import boson_basis_1d, boson_basis_general


def test_boson_1d_multidigit_roundtrip():
    # sps = Nb + 1 = 11 > 10
    basis = boson_basis_1d(L=4, Nb=10)

    # The non-bracket string representation must be injective ...
    strings = [basis.int_to_state(s, bracket_notation=False) for s in basis.states]
    assert len(set(strings)) == basis.Ns

    # ... and round-trip exactly through state_to_int() and index().
    for i, s in enumerate(basis.states):
        string = basis.int_to_state(s, bracket_notation=False)
        assert basis.state_to_int(string) == int(s)
        assert int(basis.index(string)) == i

    # The all-on-one-site state |10,0,0,0> is the base-11 number 'A000'
    # (A == 10), not the ambiguous decimal run "10000".
    assert basis.int_to_state(basis.states[0], bracket_notation=False) == "A000"


def test_boson_1d_small_sps_unchanged():
    # sps = 3 <= 10: behaviour is identical to before (plain decimal digits).
    basis = boson_basis_1d(L=3, Nb=2)
    assert [basis.int_to_state(s, False) for s in basis.states] == [
        "200",
        "110",
        "101",
        "020",
        "011",
        "002",
    ]


def test_states_that_previously_collided_are_distinct():
    # sps = 12. Under bare concatenation (11,0,0,0) and (1,10,0,0) both rendered
    # as "11000"; base-sps digits give 'B000' and '1A00' respectively.
    basis = boson_basis_1d(L=4, Nb=11)
    assert basis.state_to_int("B000") != basis.state_to_int("1A00")
    assert basis.int_to_state(basis.states[0], False) == "B000"


def test_boson_general_multidigit_roundtrip():
    basis = boson_basis_general(N=4, Nb=10, sps=11)
    for s in basis.states:
        string = basis.int_to_state(s, bracket_notation=False)
        assert basis.state_to_int(string) == int(s)


if __name__ == "__main__":
    test_boson_1d_multidigit_roundtrip()
    test_boson_1d_small_sps_unchanged()
    test_states_that_previously_collided_are_distinct()
    test_boson_general_multidigit_roundtrip()
    print("all tests passed")
