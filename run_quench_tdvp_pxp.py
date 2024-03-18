import numpy as np
import model
import tenpy
from tenpy.algorithms import tdvp, dmrg
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
import tenpy.linalg.np_conserved as npc
import os, os.path
import argparse
import logging.config
import h5py
from tenpy.tools import hdf5_io
import copy

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def coherent_state(L, p1, p2):

    T1 = np.zeros((2,1,3), dtype='complex_')
    Ta = np.zeros((2,3,3), dtype='complex_')
    Tb = np.zeros((2,3,3), dtype='complex_')
    TL = np.zeros((2,3,1), dtype='complex_')

    T1[0,0,0] = 1.
    T1[1,0,0] = p1
    
    Ta[0,0,0] = 1.
    Ta[1,0,0] = p1
    Ta[0,1,1] = 1.
    Ta[0,2,2] = 1.
    
    Tb[0,0,0] = 1.
    Tb[0,0,1] = 1.
    Tb[1,1,2] = p2
    Tb[0,2,0] = 1.
    
    TL[0,0,0] = 1.
    TL[0,2,0] = 1.
    
    T1A = npc.Array.from_ndarray_trivial(T1, labels=['p','vL','vR'], dtype='complex_')
    TaA = npc.Array.from_ndarray_trivial(Ta, labels=['p','vL','vR'], dtype='complex_')
    TbA = npc.Array.from_ndarray_trivial(Tb, labels=['p','vL','vR'], dtype='complex_')
    TLA = npc.Array.from_ndarray_trivial(TL, labels=['p','vL','vR'], dtype='complex_')

    tensors = [TaA, TbA] * L
    tensors[0] = T1A
    tensors[2*L-1] = TLA
    SVs = [np.ones(3)] * (2*L+1)  # Singular values of the tensors

    # Define the sites (assuming a spin-1/2 chain)
    sites = [SpinHalfSite(conserve=None) for _ in range(2*L)]

    # Create the MPS
    psi = MPS(sites, tensors, SVs)
    psi.canonical_form()
    return psi


if __name__ == "__main__":
    
    current_directory = os.getcwd()

    conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
    }
    logging.config.dictConfig(conf)

    parser=argparse.ArgumentParser()
    parser.add_argument("--L", default='10', help="Length of chain")
    parser.add_argument("--J0", default='1.0', help="Dipolar hopping amplitude (before quench)")
    parser.add_argument("--J", default='1.0', help="Dipolar hopping amplitude (after quench)")
    parser.add_argument("--U0", default='100.0', help="On-site Hubbard interaction (before quench)")
    parser.add_argument("--U", default='1.0', help="On-site Hubbard interaction (after quench)")
    parser.add_argument("--chi0", default='50', help="Bond dimension (before quench)")
    parser.add_argument("--chi", default='500', help="Bond dimension (after quench)")
    parser.add_argument("--Ntot", default='10', help="Total time steps")
    parser.add_argument("--dt", default='0.1', help="Delta time")
    parser.add_argument("--p", default='1.0', help="Fugacity")
    parser.add_argument("--init_state", default='2', help="Initial state")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    parser.add_argument("--p1", default='1.0', help="fugacity of 131 state")
    parser.add_argument("--p2", default='1.0', help="fugacity of 202 state")
    args = parser.parse_args()

    L = int(args.L)
    J0 = float(args.J0)
    J = float(args.J)
    U0 = float(args.U0)
    U = float(args.U)
    chi0 = int(args.chi0)
    chi = int(args.chi)
    Ntot = int(args.Ntot)
    dt = float(args.dt)
    p1 = float(args.p1)
    p2 = float(args.p2)
    init_state = args.init_state
    path = args.path
    
    #################
    # before quench #
    #################
    
    model_params0 = {
    "L": L, 
    "J": J0,
    "U": U0,
    }

    # dmrg parameters
    dmrg_params = {
    'mixer' : dmrg.SubspaceExpansion,
    'mixer_params': {
        'amplitude': 1.e-2,
        'decay': 2.0,
        'disable_after': 40
    },
    'trunc_params': {
        'chi_max': chi0,
        'svd_min': 1.e-9
    },
    'chi_list': { 0: 10, 5: 20, 10: chi0 },
    'max_E_err': 1.0e-9,
    'max_S_err': 1.0e-9,
    'max_sweeps': 50,
    'combine' : True
    }

    # initial state
    product_state1 = ["up", "up"] * int(L)
    product_state2 = ["up", "up"] * int(L)
    product_state3 = ["up", "up"] * int(L)

    product_state2[L] = "down"
    product_state3[L+1] = "down"
       
    DBHM0 = model.EFFECTIVE_PXP2(model_params0)
    psi = MPS.from_product_state(DBHM0.lat.mps_sites(), product_state1, bc=DBHM0.lat.bc_MPS)
    cdw_state = psi.copy()

    ex_131_state = MPS.from_product_state(DBHM0.lat.mps_sites(), product_state2, bc=DBHM0.lat.bc_MPS)
    ex_202_state = MPS.from_product_state(DBHM0.lat.mps_sites(), product_state3, bc=DBHM0.lat.bc_MPS)
    c_state = coherent_state(L, p1=p1, p2=p2)
    
    eng = dmrg.TwoSiteDMRGEngine(psi, DBHM0, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    EE = psi.entanglement_entropy()
    psi.canonical_form()    
    psi0 = psi.copy()

    F_CDW = np.abs(psi.overlap(cdw_state))
    F_131 = np.abs(psi.overlap(ex_131_state))
    F_202 = np.abs(psi.overlap(ex_202_state))
    F_COH = np.abs(psi.overlap(c_state))

    file = open(path+"/observables.txt","a", 1)    
    file.write(repr(0.) + " " + repr(np.max(EE)) + " " + repr(EE[len(EE)//2]) + " " + repr(1.0) + " " + repr(F_CDW) + " " + repr(F_131) + " " + repr(F_202) + " " + repr(F_COH) + " " + "\n")
    
    ################
    # after quench #
    ################

    model_params = {
    "L": L, 
    "J": J,
    "U": U,
    }

    tdvp_params = {
        'start_time': 0,
        'dt': dt,
        'N_steps': 1,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-8,
            'trunc_cut': None
        }
    }

    DBHM = model.EFFECTIVE_PXP2(model_params)    
    tdvp_engine = tdvp.TwoSiteTDVPEngine(psi, DBHM, tdvp_params)
    tdvp_two_site = True
    
    for i in range(Ntot):

        tdvp_engine.run()
        if tdvp_two_site and np.mean(psi.chi) > chi*0.85:
            tdvp_engine = tdvp.SingleSiteTDVPEngine.switch_engine(tdvp_engine)
            tdvp_two_site = False

        EE = psi.entanglement_entropy()

        F = np.abs(psi.overlap(psi0))
        F_CDW = np.abs(psi.overlap(cdw_state))
        F_131 = np.abs(psi.overlap(ex_131_state))
        F_202 = np.abs(psi.overlap(ex_202_state))
        F_COH = np.abs(psi.overlap(c_state))
        
        file.write(repr(tdvp_engine.evolved_time) + " " + repr(np.max(EE)) + " " + repr(EE[len(EE)//2]) + " " + repr(F) + " " + repr(F_CDW) + " " + repr(F_131) + " " + repr(F_202) + " " + repr(F_COH) + " " + "\n")
    
    file.close()
    