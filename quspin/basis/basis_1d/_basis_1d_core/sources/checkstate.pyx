


cdef int CheckState_P_template(int p, basis_type s, int L, basis_type[:] bitop_pars):
    cdef basis_type t=s
    cdef NP_INT8_t sign=1
    t=fliplr(t,L,&sign,bitop_pars)
    
    if t < s: 
        return 2
    elif t == s:
        if p*sign != -1:
            return 4
        else:
            return -1
    else:
        return -1


cdef int CheckState_PZ_template(int pz,basis_type s,int L, basis_type[:] bitop_pars):
    cdef basis_type t=s
    cdef NP_INT8_t sign=1
    
    t=fliplr(t,L,&sign,bitop_pars)
    t=flip_all(t,L,&sign,bitop_pars)
    if t < s:
        return 2
    elif t == s:
        if pz*sign != -1:
            return 4
        else:
            return -1
    else:
        return -1


cdef int CheckState_Z_template(int z,basis_type s,int L, basis_type[:] bitop_pars):
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    t=flip_all(t,L,&sign,bitop_pars)
    
    if t < s: 
        return 2
    elif t == s:
        if z*sign != -1:
            return 4
        else:
            return -1
    else:
        return -1


cdef int CheckState_P_Z_template(int p,int z,basis_type s,int L, basis_type[:] bitop_pars):
    cdef int rp,rz,rps

    rz = CheckState_Z_template(z,s,L,bitop_pars)
    if rz < 0:
        return -1

    rp = CheckState_P_template(p,s,L,bitop_pars)
    if rp < 0:
        return -1

    rpz = CheckState_PZ_template(z*p,s,L,bitop_pars)
    if rpz < 0:
        return -1    

    if rz == 4 and rp == 4: 
        return 16
    elif rz == 4 or rp == 4 or rpz == 4: 
        return 8
    else:
        return 4


cdef int CheckState_ZA_template(int zA,basis_type s,int L, basis_type[:] bitop_pars):
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    t=flip_sublat_A(t,L,&sign,bitop_pars)
    if t < s: 
        return 2
    elif t == s:
        if zA*sign != -1:
            return 4
        else:
            return -1
    else:
        return -1


cdef int CheckState_ZB_template(int zB,basis_type s,int L, basis_type[:] bitop_pars):
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    t=flip_sublat_B(t,L,&sign,bitop_pars)
    if t < s: 
        return 2
    elif t == s:
        if zB*sign != -1:
            return 4
        else:
            return -1
    else:
        return -1


cdef int CheckState_ZA_ZB_template(int zA,int zB,basis_type s,int L, basis_type[:] bitop_pars):
    cdef int rA,rB,rAB
    cdef NP_INT8_t sign=1

    rA = CheckState_ZA_template(zA,s,L,bitop_pars)
    if rA < 0:
        return -1

    rB = CheckState_ZB_template(zB,s,L,bitop_pars)
    if rB < 0:
        return -1

    rAB = CheckState_Z_template(zA*zB,s,L,bitop_pars)
    if rAB < 0:
        return -1    

    if rA == 4 and rB == 4:
        return 16
    elif rA == 4 or rB == 4 or rAB == 4:
        return 8
    else:
        return 4


cdef int CheckState_T_template(int kblock,int L,basis_type s,int T, basis_type[:] bitop_pars):
    # this is a function defined in [1]
    # It is used to check if the integer inputed is a reference state for a state with momentum k.
    #        kblock: the number associated with the momentum (i.e. k=2*pi*kblock/L)
    #        L: length of the system
    #        s: integer which represents a spin config in Sz basis
    #        T: number of sites to translate by, not 1 if the unit cell on the lattice has 2 sites in it.
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    cdef int R=-1
    cdef int i
    for i in range(1,L/T+1):
        t = shift(t,-T,L,&sign,bitop_pars)
        if t > s:
            return R
        elif t == s:
            if sign > 0:
                if (kblock*T*i) % L != 0: 
                    return R # need to check the shift condition
            else:
                if ((2*T*i*kblock)/L) % 2 == 0 or (2*kblock*T*i) % L != 0: 
                    return R
            R = i
            return R            


cdef void CheckState_T_P_template(int kblock,int L,basis_type s,int T,int *R, basis_type[:] bitop_pars):
    # this is a function defined in [1]
    # It is used to check if the integer inputed is a reference state for a state with momentum k.
    #        kblock: the number associated with the momentum (i.e. k=2*pi*kblock/L)
    #        L: length of the system
    #        s: integer which represents a spin config in Sz basis
    #        T: number of sites to translate by, not 1 if the unit cell on the lattice has 2 sites in it.
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    cdef int i,r
    R[0] = -1
    R[1] = -1
    R[2] = 1
    r = 0

    if CheckState_P_template(1,s,L,bitop_pars) < 0 and \
       CheckState_P_template(-1,s,L,bitop_pars) < 0:
        return
      

    for i in range(1,L/T+1):
        t = shift(t,-T,L,&sign,bitop_pars) 
        if t > s:
            return
        elif t==s:
            if sign > 0:
                if (kblock*T*i) % L != 0: 
                    return
            else:
                if ((2*T*i*kblock)/L) % 2 == 0 or (2*kblock*T*i) % L != 0: 
                    return
            R[0] = i
            r = i
            break

    t = s
    sign = 1
    t = fliplr(t,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[1] = i
            R[2] = sign
            break
        t = shift(t,-T,L,&sign,bitop_pars) 

    return


cdef void CheckState_T_P_Z_template(int kblock,int L,basis_type s,int T, int *R, basis_type[:] bitop_pars):
    # this is a function defined in [1]
    # It is used to check if the integer inputed is a reference state for a state with momentum k.
    #        kblock: the number associated with the momentum (i.e. k=2*pi*kblock/L)
    #        L: length of the system
    #        s: integer which represents a spin config in Sz basis
    #        T: number of sites to translate by, not 1 if the unit cell on the lattice has 2 sites in it.
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    R[0] = -1
    R[1] = -1
    R[2] = -1
    R[3] = -1
    R[4] =  1
    R[5] =  1
    R[6] =  1

    cdef int i,r
    r = L

    if CheckState_P_Z_template(1,1,s,L,bitop_pars) < 0 and \
       CheckState_P_Z_template(-1,1,s,L,bitop_pars) < 0  and \
       CheckState_P_Z_template(1,-1,s,L,bitop_pars) < 0 and \
       CheckState_P_Z_template(-1,-1,s,L,bitop_pars) < 0:
        return



    for i in range(1,L/T+1):
        t = shift(t,-T,L,&sign,bitop_pars)
        if t > s:
            R[0] = -1
            return
        elif t==s:
            if sign > 0:
                if (kblock*T*i) % L != 0: 
                    return
            else:
                if ((2*T*i*kblock)/L) % 2 == 0 or (2*kblock*T*i) % L != 0: 
                    return
            R[0] = i
            r = i
            break    

    t = s
    sign = 1
    t = fliplr(t,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[1] = i
            R[4] = sign
            break
        t = shift(t,-T,L,&sign,bitop_pars) 

    t = s
    sign = 1
    t = flip_all(t,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[2] = i
            R[5] = sign
            break
        t = shift(t,-T,L,&sign,bitop_pars)

    t = s
    sign = 1
    t = flip_all(t,L,&sign,bitop_pars)
    t = fliplr(t,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[3] = i
            R[6] = sign
            break
        t = shift(t,-T,L,&sign,bitop_pars)    

    return


cdef void CheckState_T_PZ_template(int kblock,int L,basis_type s,int T,int *R, basis_type[:] bitop_pars):
    # this is a function defined in [1]
    # It is used to check if the integer inputed is a reference state for a state with momentum k.
    #        kblock: the number associated with the momentum (i.e. k=2*pi*kblock/L)
    #        L: length of the system
    #        s: integer which represents a spin config in Sz basis
    #        T: number of sites to translate by, not 1 if the unit cell on the lattice has 2 sites in it.
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    R[0] = -1
    R[1] = -1 
    R[2] =  1
    cdef int i,r

    if CheckState_PZ_template(1,s,L,bitop_pars) < 0 and \
       CheckState_PZ_template(-1,s,L,bitop_pars) < 0:
        return

    r = L
    for i in range(1,L/T+1):
        t = shift(t,-T,L,&sign,bitop_pars)
        if t > s:
            return
        elif t==s:
            if sign > 0:
                if (kblock*T*i) % L != 0: 
                    return
            else:
                if ((2*T*i*kblock)/L) % 2 == 0 or (2*kblock*T*i) % L != 0: 
                    return
            R[0] = i
            r = i
            break

    t = s
    sign = 1
    t = flip_all(t,L,&sign,bitop_pars)
    t = fliplr(t,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[1] = i
            R[2] = sign
            break
        t = shift(t,-T,L,&sign,bitop_pars)

    return


cdef void CheckState_T_Z_template(int kblock,int L,basis_type s,int T,int *R, basis_type[:] bitop_pars):
    # this is a function defined in [1]
    # It is used to check if the integer inputed is a reference state for a state with momentum k.
    #        kblock: the number associated with the momentum (i.e. k=2*pi*kblock/L)
    #        L: length of the system
    #        s: integer which represents a spin config in Sz basis
    #        T: number of sites to translate by, not 1 if the unit cell on the lattice has 2 sites in it.
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    R[0] = -1
    R[1] = -1
    R[2] = 1
    cdef int i,r
    if CheckState_Z_template(1,s,L,bitop_pars) < 0 and \
       CheckState_Z_template(-1,s,L,bitop_pars) < 0:
        return

    r = L
    for i in range(1,L/T+1):
        t = shift(t,-T,L,&sign,bitop_pars)
        if t > s:
            return
        elif t==s:
            if sign > 0:
                if (kblock*T*i) % L != 0: 
                    return
            else:
                if ((2*T*i*kblock)/L) % 2 == 0 or (2*kblock*T*i) % L != 0: 
                    return
            R[0] = i
            r = i
            break

    sign = 1
    t = s
    t = flip_all(t,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[1] = i
            R[2] = sign
            break
        t = shift(t,-T,L,&sign,bitop_pars) 

    return


cdef void CheckState_T_ZA_template(int kblock,int L,basis_type s,int T,int *R, basis_type[:] bitop_pars):
    # this is a function defined in [1]
    # It is used to check if the integer inputed is a reference state for a state with momentum k.
    #        kblock: the number associated with the momentum (i.e. k=2*pi*kblock/L)
    #        L: length of the system
    #        s: integer which represents a spin config in Sz basis
    #        T: number of sites to translate by, not 1 if the unit cell on the lattice has 2 sites in it.
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    R[0] = -1
    R[1] = -1
    cdef int i,r
    if CheckState_ZA_template(1,s,L,bitop_pars) < 0 and \
       CheckState_ZA_template(-1,s,L,bitop_pars) < 0:
        return

    r = L
    for i in range(1,L/T+1):
        t = shift(t,-T,L,&sign,bitop_pars)
        if t > s:
            return
        elif t==s:
            if kblock*T*i % L != 0: # need to check the shift condition 
                return
            R[0] = i
            r = i
            break

    t = s
    t = flip_sublat_A(t,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[1] = i
            break
        t = shift(t,-T,L,&sign,bitop_pars) 

    return


cdef void CheckState_T_ZB_template(int kblock,int L,basis_type s,int T,int *R, basis_type[:] bitop_pars):
    # this is a function defined in [1]
    # It is used to check if the integer inputed is a reference state for a state with momentum k.
    #        kblock: the number associated with the momentum (i.e. k=2*pi*kblock/L)
    #        L: length of the system
    #        s: integer which represents a spin config in Sz basis
    #        T: number of sites to translate by, not 1 if the unit cell on the lattice has 2 sites in it.
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    R[0] = -1
    R[1] = -1
    cdef int i,r
    if CheckState_ZB_template(1,s,L,bitop_pars) < 0 and \
       CheckState_ZB_template(-1,s,L,bitop_pars) < 0:
        return

    r = L
    for i in range(1,L/T+1):
        t = shift(t,-T,L,&sign,bitop_pars)
        if t > s:
            return
        elif t==s:
            if kblock*T*i % L != 0: # need to check the shift condition 
                return
            R[0] = i
            r = i
            break

    t = s
    t = flip_sublat_B(t,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[1] = i
            break
        t = shift(t,-T,L,&sign,bitop_pars) 

    return


cdef void CheckState_T_ZA_ZB_template(int kblock,int L,basis_type s,int T,int *R, basis_type[:] bitop_pars):
    # this is a function defined in [1]
    # It is used to check if the integer inputed is a reference state for a state with momentum k.
    #        kblock: the number associated with the momentum (i.e. k=2*pi*kblock/L)
    #        L: length of the system
    #        s: integer which represents a spin config in Sz basis
    #        T: number of sites to translate by, not 1 if the unit cell on the lattice has 2 sites in it.
    cdef basis_type t=s
    cdef NP_INT8_t sign=1

    R[0] = -1
    R[1] = -1
    R[2] = -1
    R[3] = -1
    cdef int i,r
    if CheckState_ZA_ZB_template(1,1,s,L,bitop_pars) < 0 and \
       CheckState_ZA_ZB_template(1,-1,s,L,bitop_pars) < 0 and \
       CheckState_ZA_ZB_template(-1,1,s,L,bitop_pars) < 0 and \
       CheckState_ZA_ZB_template(-1,-1,s,L,bitop_pars) < 0:
        return


    r = L
    for i in range(1,L/T+1):
        t = shift(t,-T,L,&sign,bitop_pars)
        if t > s:
            return
        elif t==s:
            if kblock*T*i % L != 0: # need to check the shift condition 
                return
            R[0] = i
            r = i
            break


    t = flip_sublat_A(s,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[1] = i
            break
        t = shift(t,-T,L,&sign,bitop_pars) 


    t = flip_sublat_B(s,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[2] = i
            break
        t = shift(t,-T,L,&sign,bitop_pars) 


    t = flip_all(s,L,&sign,bitop_pars)
    for i in range(r):
        if t > s:
            R[0] = -1
            return
        elif t == s:
            R[3] = i
            break
        t = shift(t,-T,L,&sign,bitop_pars) 

    return



