# Copyright 2023 Hyun-Yong Lee

import numpy as np
from tenpy.networks.site import Site
from tenpy.linalg import np_conserved as npc


def BosonSite_DM_conserved(Nmax=2, cons_N='N', cons_D='D', x=0):
    
    dim = Nmax + 1
    states = [str(n) for n in range(0, dim)]
    if dim < 2:
       raise ValueError("local dimension should be larger than 1....")
    
    # 0) define the operators
    B = np.zeros([dim, dim], dtype=np.float64)  # destruction/annihilation operator
    for n in range(1, dim):
       B[n - 1, n] = np.sqrt(n)
    Bd = np.transpose(B)  # .conj() wouldn't do anything
    # Note: np.dot(Bd, B) has numerical roundoff errors of eps~=4.4e-16.
    Ndiag = np.arange(dim, dtype=np.float64)
    N = np.diag(Ndiag)
    NN = np.diag(Ndiag**2)
    dN = np.diag(Ndiag)
    dNdN = np.diag((Ndiag)**2)
    P = np.diag(1. - 2. * np.mod(Ndiag, 2))
    ops = dict(B=B, Bd=Bd, N=N, NN=NN, dN=dN, dNdN=dNdN, P=P)

    # 1) handle charges
    qmod = []
    qnames = []
    charges = []
    if cons_N == 'N':
        qnames.append('N')
        qmod.append(1)
        charges.append( [i for i in range(dim)] )

    if cons_D == 'D':
        qnames.append('D')  # Dipole moment
        qmod.append(1)
        charges.append( [i*x for i in range(dim)] )

    if len(qmod) == 1:
        charges = charges[0]
    else:  # len(charges) == 2: need to transpose
        charges = [[q1, q2] for q1, q2 in zip(charges[0], charges[1])]
    chinfo = npc.ChargeInfo(qmod, qnames)
    leg = npc.LegCharge.from_qflat(chinfo, charges)

    return Site(leg, states, sort_charge=True, **ops)
