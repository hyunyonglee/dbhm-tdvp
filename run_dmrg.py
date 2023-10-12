import numpy as np
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms import tebd
from tenpy.tools.process import mkl_set_nthreads
import argparse
import logging.config
import os
import os.path
import h5py
from tenpy.tools import hdf5_io
import model

mkl_set_nthreads(64)

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def measurements(psi, L):
    
    # Measurements
    Ns = psi.expectation_value("N")
    NNs = psi.expectation_value("NN")
    EE = psi.entanglement_entropy()
    
    return Ns, NNs, EE


def write_data( psi, E, Ns, NNs, EE, L, Ncut, td, U, path ):

    ensure_dir(path+"/observables/")
    ensure_dir(path+"/mps/")

    data = {"psi": psi}
    with h5py.File(path+"/mps/psi_L_%d_Ncut_%d_td_%.2f_U_%.2f.h5" % (L,Ncut,td,U), 'w') as f:
        hdf5_io.save_to_hdf5(f, data)

    file_EE = open(path+"/observables/EE.txt","a", 1)    
    file_Ns = open(path+"/observables/Ns.txt","a", 1)
    file_NNs = open(path+"/observables/NNs.txt","a", 1)
    
    file_EE.write(repr(td) + " " + repr(U) + " " + "  ".join(map(str, EE)) + " " + "\n")
    file_Ns.write(repr(td) + " " + repr(U) + " " + "  ".join(map(str, Ns)) + " " + "\n")
    file_NNs.write(repr(td) + " " + repr(U) + " " + "  ".join(map(str, NNs)) + " " + "\n")
    
    file_EE.close()
    file_Ns.close()
    file_NNs.close()
    
    #
    file = open(path+"/observables.txt","a", 1)    
    file.write(repr(td) + " " + repr(U) + " " + repr(E) + " " + repr(np.max(EE)) + " " + repr(np.mean(Ns)) + " " + repr(np.mean(NNs)) + " " + "\n")
    file.close()
    

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
    parser.add_argument("--td", default='0.0', help="Dipolar hopping amplitude")
    parser.add_argument("--U", default='1.0', help="On-site Hubbard interaction")
    parser.add_argument("--chi", default='64', help="Bond dimension")
    parser.add_argument("--Ncut", default='4', help="Cut-off boson number")
    parser.add_argument("--init_state", default='2', help="Initial state")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    args=parser.parse_args()

    L = int(args.L)
    td = float(args.td)
    U = float(args.U)
    chi = int(args.chi)
    Ncut = int(args.Ncut)
    init_state = args.init_state
    path = args.path
    
    model_params = {
    "L": L, 
    "td": td,
    "U": U,
    "Ncut": Ncut
    }

    DBHM = model.DIPOLAR_BOSE_HUBBARD_CONSERVED(model_params)

    # initial state
    if init_state == '1':
        product_state = ['1'] * L
    elif init_state == '2':
        product_state = ['2'] * L
    elif init_state == '1-half-a':
        product_state = ['1','2'] * int(L/2)
    elif init_state == '1-half-b':
        product_state = ['1','1','2','2'] * int(L/4)
    elif init_state == '2-half-a':
        product_state = ['2','3'] * int(L/2)
    elif init_state == '2-half-b':
        product_state = ['2','2','3','3'] * int(L/4)
    
    psi = MPS.from_product_state(DBHM.lat.mps_sites(), product_state, bc=DBHM.lat.bc_MPS)

    dmrg_params = {
    # 'mixer': True,  # setting this to True helps to escape local minima
    'mixer' : dmrg.SubspaceExpansion,
    'mixer_params': {
        'amplitude': 1.e-4,
        'decay': 2.0,
        'disable_after': 20
    },
    'trunc_params': {
        'chi_max': chi,
        'svd_min': 1.e-9
    },
    'chi_list': { 0: 8, 5: 16, 10: 32, 15: 64, 20: chi },
    'max_E_err': 1.0e-9,
    'max_S_err': 1.0e-9,
    'max_sweeps': 100,
    'combine' : True
    }

    # ground state
    eng = dmrg.TwoSiteDMRGEngine(psi, DBHM, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    psi.canonical_form() 

    Ns, NNs, EE = measurements(psi, L)
    write_data( psi, E, Ns, NNs, EE, L, Ncut, td, U, path )