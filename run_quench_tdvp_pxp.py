import numpy as np
import model
from tenpy.algorithms import tdvp, dmrg
from tenpy.networks.mps import MPS
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
    parser.add_argument("--init_state", default='2', help="Initial state")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
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
        'decay': 1.3,
        'disable_after': 100
    },
    'trunc_params': {
        'chi_max': chi0,
        'svd_min': 1.e-9
    },
    'chi_list': { 0: 10, 5: 20, 10: chi0 },
    'max_E_err': 1.0e-9,
    'max_S_err': 1.0e-9,
    'max_sweeps': 100,
    'combine' : True
    }

    # initial state
    product_state1 = []
    product_state2 = []
    product_state3 = []
    local_state1 = np.array( [1., 0., 0., 0.] )
    for i in range(0, L):
        product_state1.append(local_state1)
        product_state2.append(local_state1)
        product_state3.append(local_state1)
    
    product_state2[L//2] = np.array( [0., 1., 0., 0.] )
    product_state3[L//2] = np.array( [0., 0., 1., 0.] )
        
    file = open(path+"/observables.txt","a", 1)    
    

    DBHM0 = model.EFFECTIVE_PXP(model_params0)
    psi = MPS.from_product_state(DBHM0.lat.mps_sites(), product_state1, bc=DBHM0.lat.bc_MPS)
    cdw_state = psi.copy()

    ex_131_state = MPS.from_product_state(DBHM0.lat.mps_sites(), product_state2, bc=DBHM0.lat.bc_MPS)
    ex_202_state = MPS.from_product_state(DBHM0.lat.mps_sites(), product_state3, bc=DBHM0.lat.bc_MPS)
    
    eng = dmrg.TwoSiteDMRGEngine(psi, DBHM0, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    EE = psi.entanglement_entropy()
    psi.canonical_form()    
    psi0 = psi.copy()

    F_CDW = np.abs(psi.overlap(cdw_state))
    F_131 = np.abs(psi.overlap(ex_131_state))
    F_202 = np.abs(psi.overlap(ex_202_state))

    file.write(repr(0.) + " " + repr(np.max(EE)) + " " + repr(EE[len(EE)//2]) + " " + repr(1.0) + " " + repr(F_CDW) + " " + repr(F_131) + " " + repr(F_202) + " " + "\n")
    
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

    DBHM = model.EFFECTIVE_PXP(model_params)    
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
        
        file.write(repr(tdvp_engine.evolved_time) + " " + repr(np.max(EE)) + " " + repr(EE[len(EE)//2]) + " " + repr(F) + " " + repr(F_CDW) + " " + repr(F_131) + " " + repr(F_202) + " " + "\n")
    
    file.close()
    