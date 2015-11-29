Basis_fortran: symmetry_ops.f90 m_basis.f90 z_basis.f90 p_basis.f90 pz_basis.f90 p_z_basis.f90 spinop.f90 refstate.f90 z_refstate.f90 p_refstate.f90 pz_refstate.f90 p_z_refstate.f90
	f2py $^ -m Basis_fortran -h Basis_fortran.pyf skip: flip_lr flip_all checkstate_z checkstate_p checkstate_pz findzstate  --overwrite-signature --quiet
	f2py -c Basis_fortran.pyf $^ --opt=-o3 --quiet
