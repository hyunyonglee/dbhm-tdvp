import numpy as np
import model
from tenpy.algorithms import tdvp, dmrg
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

def flip_array(array):
    # 배열의 길이
    length = len(array)
    # 1/3 지점과 2/3 지점 계산
    I = length // 3
    J = 2 * I
    
    # 1/3 지점에서 요소 뒤집기
    if array[I:I+2] == ['2', '1']:
        array[I:I+2] = ['1', '2']
        if array[J:J+2] == ['1', '2']:
            array[J:J+2] = ['2', '1']
        else:
            array[(J-1):J+1] = ['2', '1']
    else:
        array[I:I+2] = ['2', '1']
        if array[J:J+2] == ['2', '1']:
            array[J:J+2] = ['1', '2']
        else:
            array[(J-1):J+1] = ['1', '2']
    return array

def measurements(psi, L, Qsp=False):
    
    # Measurements
    Ns = psi.expectation_value("N")
    NNs = psi.expectation_value("NN")
    EE = psi.entanglement_entropy()
    
    # Measuring Correlation functions from the center
    Cnn_center = np.zeros(L)
    Density_center = np.zeros(L-1)
    Dsp_center1 = np.zeros(L-1)
    Dsp_center2 = np.zeros(L-1)
    
    Qsp_center1 = np.zeros(L-2)
    Qsp_center2 = np.zeros(L-2)
    
    for i in range(0,L):
        I = i
        if L%2 == 0:
            J = int(L/2)-1
        else:
            J = int(L/2)
        
        C = psi.expectation_value_term([('N',I),('N',J)])
        C = C - psi.expectation_value_term([('N',I)]) * psi.expectation_value_term([('N',J)])
        Cnn_center[i] = C.real
        
        if i<L-1:
            DensityC = psi.expectation_value_term([('N',I),('N',I+1),('N',J),('N',J+1)])
            DensityC = DensityC - psi.expectation_value_term([('N',I),('N',I+1)]) * psi.expectation_value_term([('N',J),('N',J+1)])
            Density_center[i] = DensityC.real
        
        if i<L-1:
            D = psi.expectation_value_term([('Bd',I),('B',I+1),('Bd',J),('B',J-1)])
            Dsp_center1[i] = D.real
            D = psi.expectation_value_term([('Bd',I),('B',I+1),('Bd',J+1),('B',J)])
            Dsp_center2[i] = D.real

        if Qsp and i<L-2:
            Q = psi.expectation_value_term([('Bd',I),('B',I+1),('B',I+1),('Bd',I+2),('B',J+1),('Bd',J),('Bd',J),('B',J-1)])
            Q = Q - psi.expectation_value_term([('Bd',I),('B',I+1),('B',I+1),('Bd',I+2)]) * psi.expectation_value_term([('B',J+1),('Bd',J),('Bd',J),('B',J-1)])
            Qsp_center1[i] = Q.real
            
            Q = psi.expectation_value_term([('Bd',I),('B',I+1),('B',I+1),('Bd',I+2),('B',J+2),('Bd',J+1),('Bd',J+1),('B',J)])
            Q = Q - psi.expectation_value_term([('Bd',I),('B',I+1),('B',I+1),('Bd',I+2)]) * psi.expectation_value_term([('B',J+2),('Bd',J+1),('Bd',J+1),('B',J)])
            Qsp_center2[i] = Q.real
        
    return Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Density_center, EE


def dc_corr_func(psi, L, time, path):
    
    # Measuring Correlation functions from the center
    Dsp_corr = np.zeros((L-1,L-1), dtype=complex)
    
    for i in range(0,L-1):
        for j in range(0,L-1):
            Dsp_corr[i,j] = psi.expectation_value_term([('Bd',i),('B',i+1),('Bd',j+1),('B',j)])
    
    ensure_dir(path+"/observables/")
    file_Dsp_corr1 = open(path+"/observables/Dsp_corr_real_t_%.3f.txt" % time,"a", 1)
    file_Dsp_corr2 = open(path+"/observables/Dsp_corr_imag_t_%.3f.txt" % time,"a", 1)
    
    # write real part of correlation function
    for i in range(0,L-1):
        file_Dsp_corr1.write("  ".join(map(str, Dsp_corr[i,:].real)) + " " + "\n")
        file_Dsp_corr2.write("  ".join(map(str, Dsp_corr[i,:].imag)) + " " + "\n")
    file_Dsp_corr1.close()
    file_Dsp_corr2.close()


def write_data( Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Density_center, Bcor, Ncor, Dcor, F, F1, F2, EE, time, path ):

    ensure_dir(path+"/observables/")
    ensure_dir(path+"/mps/")

    # data = {"psi": psi}
    # with h5py.File(path+"/mps/psi_time_%.3f.h5" % time, 'w') as f:
    #     hdf5_io.save_to_hdf5(f, data)

    file_EE = open(path+"/observables/EE.txt","a", 1)    
    file_Ns = open(path+"/observables/Ns.txt","a", 1)
    file_NNs = open(path+"/observables/NNs.txt","a", 1)
    file_Cnn = open(path+"/observables/Cnn.txt","a", 1)
    file_Density = open(path+"/observables/DensityDensity.txt","a", 1)
    file_Dsp1 = open(path+"/observables/Dsp1.txt","a", 1)
    file_Dsp2 = open(path+"/observables/Dsp2.txt","a", 1)
    file_Qsp1 = open(path+"/observables/Qsp1.txt","a", 1)
    file_Qsp2 = open(path+"/observables/Qsp2.txt","a", 1)

    file_EE.write(repr(time) + " " + "  ".join(map(str, EE)) + " " + "\n")
    file_Ns.write(repr(time) + " " + "  ".join(map(str, Ns)) + " " + "\n")
    file_NNs.write(repr(time) + " " + "  ".join(map(str, NNs)) + " " + "\n")
    file_Cnn.write(repr(time) + " " + "  ".join(map(str, Cnn_center)) + " " + "\n")
    file_Density.write(repr(time) + " " + "  ".join(map(str, Density_center)) + " " + "\n")
    file_Dsp1.write(repr(time) + " " + "  ".join(map(str, Dsp_center1)) + " " + "\n")
    file_Dsp2.write(repr(time) + " " + "  ".join(map(str, Dsp_center2)) + " " + "\n")
    file_Qsp1.write(repr(time) + " " + "  ".join(map(str, Qsp_center1)) + " " + "\n")
    file_Qsp2.write(repr(time) + " " + "  ".join(map(str, Qsp_center2)) + " " + "\n")
    
    file_EE.close()
    file_Ns.close()
    file_NNs.close()
    file_Cnn.close()
    file_Density.close()
    file_Dsp1.close()
    file_Dsp2.close()
    file_Qsp1.close()
    file_Qsp2.close()
    
    #
    file = open(path+"/observables.txt","a", 1)    
    file.write(repr(time) + " " + repr(np.max(EE)) + " " + repr(np.mean(Ns)) + " " + repr(np.mean(NNs)) + " " + repr(np.abs(Bcor)) + " " + repr(np.abs(Ncor)) + " " + repr(np.abs(Dcor)) + " " + repr(np.abs(F)) + " " + repr(F.real) + " " + repr(F.imag) + " " + repr(np.abs(F1)) + " " + repr(np.abs(F2)) + " " + "\n")
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
    parser.add_argument("--td0", default='1.0', help="Dipolar hopping amplitude (before quench)")
    parser.add_argument("--td", default='1.0', help="Dipolar hopping amplitude (after quench)")
    parser.add_argument("--U0", default='100.0', help="On-site Hubbard interaction (before quench)")
    parser.add_argument("--U", default='1.0', help="On-site Hubbard interaction (after quench)")
    parser.add_argument("--chi0", default='50', help="Bond dimension (before quench)")
    parser.add_argument("--chi", default='500', help="Bond dimension (after quench)")
    parser.add_argument("--Ncut", default='4', help="Cut-off boson number")
    parser.add_argument("--Ntot", default='10', help="Total time steps")
    parser.add_argument("--Mstep", default='5', help="Measurement time step")
    parser.add_argument("--Pstep", default='1000', help="Wavefunction save time step")
    parser.add_argument("--dt", default='0.1', help="Delta time")
    parser.add_argument("--init_state", default='2', help="Initial state")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    parser.add_argument('--autocorr', action='store_true', help='enable autocorrelation function calculation')
    parser.add_argument('--d_corr_func', action='store_true', help='enable dipole correlation function calculation')
    parser.add_argument('--q_corr_func', action='store_true', help='enable quadrupole correlation function calculation')
    args = parser.parse_args()

    L = int(args.L)
    td0 = float(args.td0)
    td = float(args.td)
    U0 = float(args.U0)
    U = float(args.U)
    chi0 = int(args.chi0)
    chi = int(args.chi)
    Ncut = int(args.Ncut)
    Ntot = int(args.Ntot)
    Mstep = int(args.Mstep)
    Pstep = int(args.Pstep)
    dt = float(args.dt)
    init_state = args.init_state
    path = args.path
    
    #################
    # before quench #
    #################
    
    model_params0 = {
    "L": L, 
    "td": td0,
    "U": U0,
    "Ncut": Ncut
    }

    # dmrg parameters
    dmrg_params = {
    'mixer' : dmrg.SubspaceExpansion,
    'mixer_params': {
        'amplitude': 1.e-4,
        'decay': 2.0,
        'disable_after': 20
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
    if init_state == '1':
        product_state = ['1'] * L
    elif init_state == '2':
        product_state = ['2'] * L
    elif init_state == '1-half-a':
        if L % 2 == 0:
            raise ValueError("Length must be odd.")
        else:
            product_state = ['1', '2'] * (L // 2) + ['1']
    elif init_state == '1-half-b':
        if L % 4 != 2:
            raise ValueError("Length must be 4n+2.")
        else:
            product_state = ['1', '1', '2', '2'] * (L // 4) + ['1','1']
    elif init_state == '3':
        product_state = ['3'] * L
    elif init_state == '4':
        product_state = ['4'] * L
    elif init_state == '5':
        product_state = ['5'] * L
    
    DBHM0 = model.DIPOLAR_BOSE_HUBBARD_CONSERVED(model_params0)
    psi = MPS.from_product_state(DBHM0.lat.mps_sites(), product_state, bc=DBHM0.lat.bc_MPS)

    product_state1 = MPS.from_product_state(DBHM0.lat.mps_sites(), product_state, bc=DBHM0.lat.bc_MPS)
    product_state2 = MPS.from_product_state(DBHM0.lat.mps_sites(), flip_array(product_state), bc=DBHM0.lat.bc_MPS)

    # ground state
    dmrg_params['orthogonal_to'] = [product_state1]
    eng = dmrg.TwoSiteDMRGEngine(product_state2, DBHM0, dmrg_params)
    
    # eng = dmrg.TwoSiteDMRGEngine(psi, DBHM0, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    psi.canonical_form()
    psi0 = psi.copy()
    
    if args.autocorr:
        # prepare for autocorrelation functions
        psi_b = psi.copy()
        psi_n = psi.copy()
        psi_d = psi.copy()

        psi_b.apply_local_op(i=int(L/2), op='B')#, unitary=False)
        psi_n.apply_local_op(i=int(L/2), op='N')#, unitary=False)
        psi_d.apply_local_op(i=(int(L/2)-1), op='Bd')#, unitary=False)
        psi_d.apply_local_op(i=int(L/2), op='B')#, unitary=False)
    
        psi_T_b = psi.copy()
        psi_T_n = psi.copy()
        psi_T_d = psi.copy()
    
        psi_T_b.apply_local_op(i=int(L/2), op='B')#, unitary=False)
        psi_T_n.apply_local_op(i=int(L/2), op='N')#, unitary=False)
        psi_T_d.apply_local_op(i=(int(L/2)-1), op='Bd')#, unitary=False)
        psi_T_d.apply_local_op(i=int(L/2), op='B')#, unitary=False)
    
        Bcor = psi_b.overlap( psi_T_b )
        Ncor = psi_n.overlap( psi_T_n )
        Dcor = psi_d.overlap( psi_T_d )
    else:
        Bcor = 0.
        Ncor = 0.
        Dcor = 0.
    
    Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Density_center, EE = measurements(psi, L, args.q_corr_func)
    F1 = psi.overlap(product_state1)
    F2 = psi.overlap(product_state2)        
    write_data( Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Density_center, Bcor, Ncor, Dcor, 1., F1, F2, EE, 0, path )

    ################
    # after quench #
    ################

    model_params = {
    "L": L, 
    "td": td,
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

    DBHM = model.DIPOLAR_BOSE_HUBBARD_CONSERVED(model_params)    
    tdvp_engine = tdvp.TwoSiteTDVPEngine(psi, DBHM, tdvp_params)
    tdvp_two_site = True
    
    if args.autocorr:
        tdvp_engine_b = tdvp.TwoSiteTDVPEngine(psi_b, DBHM, tdvp_params)
        tdvp_engine_n = tdvp.TwoSiteTDVPEngine(psi_n, DBHM, tdvp_params)
        tdvp_engine_d = tdvp.TwoSiteTDVPEngine(psi_d, DBHM, tdvp_params)

        tdvp_two_site_b = True
        tdvp_two_site_n = True
        tdvp_two_site_d = True

    for i in range(Ntot):

        tdvp_engine.run()
        if tdvp_two_site and np.mean(psi.chi) > chi*0.85:
            tdvp_engine = tdvp.SingleSiteTDVPEngine.switch_engine(tdvp_engine)
            tdvp_two_site = False

        
        if args.autocorr:
            tdvp_engine_b.run()
            tdvp_engine_n.run()
            tdvp_engine_d.run()

            if tdvp_two_site_b and np.mean(psi_b.chi) > chi*0.85:
                tdvp_engine_b = tdvp.SingleSiteTDVPEngine.switch_engine(tdvp_engine_b)
                tdvp_two_site_b = False

            if tdvp_two_site_n and np.mean(psi_n.chi) > chi*0.85:
                tdvp_engine_n = tdvp.SingleSiteTDVPEngine.switch_engine(tdvp_engine_n)
                tdvp_two_site_n = False

            if tdvp_two_site_d and np.mean(psi_d.chi) > chi*0.85:
                tdvp_engine_d = tdvp.SingleSiteTDVPEngine.switch_engine(tdvp_engine_d)
                tdvp_two_site_d = False

        if (i+1) % Mstep == 0:    
            Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Density_center, EE = measurements(psi, L, args.q_corr_func)

            if args.autocorr:
                psi_T_b = psi.copy()
                psi_T_n = psi.copy()
                psi_T_d = psi.copy()
            
                psi_T_b.apply_local_op(i=int(L/2), op='B')#, unitary=False)
                psi_T_n.apply_local_op(i=int(L/2), op='N')#, unitary=False)
                psi_T_d.apply_local_op(i=(int(L/2)-1), op='Bd')#, unitary=False)
                psi_T_d.apply_local_op(i=int(L/2), op='B')#, unitary=False)

                Bcor = psi_b.overlap( psi_T_b )
                Ncor = psi_n.overlap( psi_T_n )
                Dcor = psi_d.overlap( psi_T_d )

            F = psi.overlap(psi0)
            F1 = psi.overlap(product_state1)
            F2 = psi.overlap(product_state2)
            write_data( Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Density_center, Bcor, Ncor, Dcor, F, F1, F2, EE, tdvp_engine.evolved_time, path )
            
            if args.d_corr_func:
                dc_corr_func(psi, L, tdvp_engine.evolved_time, path)

        if (i+1) % Pstep == 0:
            
            data = {"psi": psi}
            with h5py.File(path+"/mps/psi_time_%.3f.h5" % tdvp_engine.evolved_time, 'w') as f:
                hdf5_io.save_to_hdf5(f, data)