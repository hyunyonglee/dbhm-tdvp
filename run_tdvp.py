import numpy as np
import model
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS
import os, os.path
import argparse
import logging.config
import h5py
from tenpy.tools import hdf5_io

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
    
    # Measuring Dipole Bound State
    Ds = []
    for i in range(0,L-1): 
        Ds.append( np.abs( psi.expectation_value_term([('Bd',i+1),('B',i)]) ) )
    
    # Measuring Single-Particle Correlation
    Csp = np.zeros((9,int(L/3)))
    Dsp = np.zeros((9,int(L/3)))
    Cnn = np.zeros((9,int(L/3)))
    Dnn = np.zeros((9,int(L/3)))
    for r in range(1,10):

        for i in range(0,int(L/3)):
            I = int(L/3)+i
            J = I+r
            
            C = psi.expectation_value_term([('Bd',I),('B',J)])
            C = C + psi.expectation_value_term([('B',I),('Bd',J)])
            Csp[r-1,i] = C.real

            D = psi.expectation_value_term([('Bd',I),('B',I+1),('B',J),('Bd',J+1)])
            D = D - psi.expectation_value_term([('Bd',I),('B',I+1)]) * psi.expectation_value_term([('B',J),('Bd',J+1)])
            D = D + psi.expectation_value_term([('B',I),('Bd',I+1),('Bd',J),('B',J+1)])
            D = D - psi.expectation_value_term([('B',I),('Bd',I+1)]) * psi.expectation_value_term([('Bd',J),('B',J+1)])
            Dsp[r-1,i] = D.real

            C = psi.expectation_value_term([('N',I),('N',J)])
            C = C - psi.expectation_value_term([('N',I)]) * psi.expectation_value_term([('N',J)])
            Cnn[r-1,i] = C.real

            D = psi.expectation_value_term([('Bd',I+1),('B',I),('Bd',I),('B',I+1), ('Bd',J+1),('B',J),('Bd',J),('B',J+1) ])
            D = D - psi.expectation_value_term([('Bd',I+1),('B',I),('Bd',I),('B',I+1)]) * psi.expectation_value_term([('Bd',J+1),('B',J),('Bd',J),('B',J+1)])
            Dnn[r-1,i] = D.real

    return Ns, NNs, Ds, Csp, Cnn, Dsp, Dnn, EE



def write_data( psi, Ns, NNs, Ds, Csp, Cnn, Dsp, Dnn, EE, time, path ):

    ensure_dir(path+"/observables/")
    ensure_dir(path+"/mps/")

    # data = {"psi": psi}
    # with h5py.File(path+"/mps/psi_time_%.3f.h5" % time, 'w') as f:
    #     hdf5_io.save_to_hdf5(f, data)

    file_EE = open(path+"/observables/EE.txt","a", 1)    
    file_Ns = open(path+"/observables/Ns.txt","a", 1)
    file_NNs = open(path+"/observables/NNs.txt","a", 1)
    file_Ds = open(path+"/observables/Ds.txt","a", 1)
    
    file_EE.write(repr(time) + " " + "  ".join(map(str, EE)) + " " + "\n")
    file_Ns.write(repr(time) + " " + "  ".join(map(str, Ns)) + " " + "\n")
    file_NNs.write(repr(time) + " " + "  ".join(map(str, NNs)) + " " + "\n")
    file_Ds.write(repr(time) + " " + "  ".join(map(str, Ds)) + " " + "\n")
    
    file_EE.close()
    file_Ns.close()
    file_NNs.close()
    file_Ds.close()
    
    # Writing Correlation functions
    for r in range(1,10):

        file_Csp = open(path+"/observables/Csp_r%d.txt" % r,"a", 1)
        file_Cnn = open(path+"/observables/Cnn_r%d.txt" % r,"a", 1)
        file_Dsp = open(path+"/observables/Dsp_r%d.txt" % r,"a", 1)
        file_Dnn = open(path+"/observables/Dnn_r%d.txt" % r,"a", 1)

        file_Csp.write(repr(time) + " " + "  ".join(map(str, Csp[r-1,:])) + " " + "\n")
        file_Cnn.write(repr(time) + " " + "  ".join(map(str, Cnn[r-1,:])) + " " + "\n")
        file_Dsp.write(repr(time) + " " + "  ".join(map(str, Dsp[r-1,:])) + " " + "\n")
        file_Dnn.write(repr(time) + " " + "  ".join(map(str, Dnn[r-1,:])) + " " + "\n")

        file_Csp.close()
        file_Cnn.close()
        file_Dsp.close()
        file_Dnn.close()
    
    #
    file = open(path+"/observables.txt","a", 1)    
    file.write(repr(time) + " " + repr(np.max(EE)) + " " + repr(np.mean(Ns)) + " " + repr(np.mean(NNs)) + " " + repr(np.mean(Ds)) 
               + " " + "  ".join(map(str, np.mean(Csp,axis=1))) + " " + "  ".join(map(str, np.mean(Cnn,axis=1))) 
               + " " + "  ".join(map(str, np.mean(Dsp,axis=1))) + " " + "  ".join(map(str, np.mean(Dnn,axis=1))) + " " + "\n")
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
    parser.add_argument("--ts", default='0.0', help="Single-particle hopping amplitude")
    parser.add_argument("--U", default='1.0', help="On-site Hubbard interaction")
    parser.add_argument("--chi", default='64', help="Bond dimension")
    parser.add_argument("--Ncut", default='4', help="Cut-off boson number")
    parser.add_argument("--Ntot", default='10', help="Total time steps")
    parser.add_argument("--Mstep", default='5', help="Measurement time step")
    parser.add_argument("--dt", default='0.1', help="Delta time")
    parser.add_argument("--init_state", default='2', help="Initial state")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    args=parser.parse_args()

    L = int(args.L)
    td = float(args.td)
    ts = float(args.ts)
    U = float(args.U)
    chi = int(args.chi)
    Ncut = int(args.Ncut)
    Ntot = int(args.Ntot)
    Mstep = int(args.Mstep)
    dt = float(args.dt)
    init_state = args.init_state
    path = args.path
    
    model_params0 = {
    "L": L, 
    "td": 0.0,
    "ts": 1.0e-5,
    "U": 0.0,
    "Ncut": Ncut
    }

    model_params = {
    "L": L, 
    "td": td,
    "ts": ts,
    "U": U,
    "Ncut": Ncut
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

    DBHM0 = model.DIPOLAR_BOSE_HUBBARD(model_params0)
    DBHM = model.DIPOLAR_BOSE_HUBBARD(model_params)

    # initial state
    if init_state == '2':
        product_state = ['2'] * DBHM0.lat.N_sites
    elif init_state == '1-half':
        product_state = ['1','2'] * int(DBHM0.lat.N_sites/2)
    elif init_state == '1-half-2':
        product_state = ['1','1','2','2'] * int(DBHM0.lat.N_sites/4)
    elif init_state == '2-half':
        product_state = ['2','3'] * int(DBHM0.lat.N_sites/2)
    elif init_state == '2-half-2':
        product_state = ['2','2','3','3'] * int(DBHM0.lat.N_sites/4)

    psi = MPS.from_product_state(DBHM0.lat.mps_sites(), product_state, bc=DBHM0.lat.bc_MPS)

    tdvp_engine0 = tdvp.TwoSiteTDVPEngine(psi, DBHM0, tdvp_params)
    for i in range(5):
        tdvp_engine0.run()
        
    Ns, NNs, Ds, Csp, Cnn, Dsp, Dnn, EE = measurements(psi, L)
    write_data( psi, Ns, NNs, Ds, Csp, Cnn, Dsp, Dnn, EE, 0, path )

    tdvp_engine = tdvp.TwoSiteTDVPEngine(psi, DBHM, tdvp_params)
    for i in range(Ntot):
        tdvp_engine.run()
        if (i+1) % Mstep == 0:
            Ns, NNs, Ds, Csp, Cnn, Dsp, Dnn, EE = measurements(psi, L)
            write_data( psi, Ns, NNs, Ds, Csp, Cnn, Dsp, Dnn, EE, tdvp_engine.evolved_time, path )