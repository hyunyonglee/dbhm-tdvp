import numpy as np
import model
import tenpy
from tenpy.algorithms import tdvp, dmrg
from tenpy.networks.mps import MPS
from tenpy.networks.site import BosonSite
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

def MPS_drop_charge(psi, charge=None, chinfo=None, permute_p_leg=True):
    psi_c = psi.copy()
    psi_c.chinfo = chinfo = npc.ChargeInfo.drop(chinfo, charge=charge)
    if permute_p_leg is None and chinfo.qnumber == 0:
        permute_p_leg = True
    for i, B in enumerate(psi_c._B):
        psi_c._B[i] = B = B.drop_charge(charge=charge, chinfo=chinfo)
        psi_c.sites[i] = site = copy.copy(psi.sites[i])
        if permute_p_leg:
            if permute_p_leg is True:
                perm = tenpy.tools.misc.inverse_permutation(site.perm)
            else:
                perm = permute_p_leg
            psi_c._B[i] = B = B.permute(perm, 'p')
        else:
            perm = None
        site.change_charge(B.get_leg('p'), perm) # in place
    print(psi_c.chinfo)
    psi_c.test_sanity()
    return psi_c

def coherent_state(L, Ncut, p1, p2):

    T1 = np.zeros((Ncut+1,1,3), dtype='complex_')
    Ta = np.zeros((Ncut+1,3,3), dtype='complex_')
    Tb = np.zeros((Ncut+1,3,3), dtype='complex_')
    TL = np.zeros((Ncut+1,3,1), dtype='complex_')

    T1[1,0,0] = 1.
    T1[2,0,1] = 1.

    Ta[1,0,0] = 1.
    Ta[2,0,1] = 1.
    Ta[2,1,0] = 1.
    Ta[3,2,2] = p1

    Tb[2,0,0] = 1.
    Tb[0,1,1] = p2
    Tb[1,2,0] = 1.
    Tb[1,0,2] = 1.

    TL[1,0,0] = 1.
    TL[2,1,0] = 1.

    T1A = npc.Array.from_ndarray_trivial(T1, labels=['p','vL','vR'], dtype='complex_')
    TaA = npc.Array.from_ndarray_trivial(Ta, labels=['p','vL','vR'], dtype='complex_')
    TbA = npc.Array.from_ndarray_trivial(Tb, labels=['p','vL','vR'], dtype='complex_')
    TLA = npc.Array.from_ndarray_trivial(TL, labels=['p','vL','vR'], dtype='complex_')

    tensors = [TaA, TbA, TaA, TbA] * (L//4) + [TLA]
    tensors[0] = T1A
    SVs = [np.ones(3)] * (L+1)  # Singular values of the tensors

    # Define the sites (assuming a spin-1/2 chain)
    sites = [BosonSite(Nmax=Ncut, conserve=None) for _ in range(L)]

    # Create the MPS
    psi = MPS(sites, tensors, SVs)
    psi.canonical_form()
    return psi

def ex_131_configuration(L):
    
    ex_state = ['1', '2'] * (L//2) + ['1']
    C = L // 2
    
    if ex_state[C] == '1':
        ex_state[C-1:C+2] = ['1','3','1']
    else:
        ex_state[C:C+3] = ['1','3','1']

    return ex_state

def ex_202_configuration(L):
    
    ex_state = ['1', '2'] * (L//2) + ['1']
    C = L // 2
    
    if ex_state[C] == '2':
        ex_state[C-1:C+2] = ['2','0','2']
    else:
        ex_state[C:C+3] = ['2','0','2']

    return ex_state

def ex_040_configuration(L):
    
    ex_state = ['1', '2'] * (L//2) + ['1']
    C = L // 2
    
    if ex_state[C] == '2':
        ex_state[C-1:C+2] = ['0','4','0']
    else:
        ex_state[C:C+3] = ['0','4','0']

    return ex_state

def lr1_configuration(L):

    lr_state = ['1', '2'] * (L//2) + ['1']
    C = (L // 2)-3
    if lr_state[C] == '1':
        lr_state[C:C+5] = ['2','1','1','1','2']
    else:
        lr_state[(C-1):C+4] = ['2','1','1','1','2']

    return lr_state

def lr2_configuration(L):
    # 배열의 길이
    lr_state = ['1', '2'] * (L//2) + ['1']
    
    # 1/3 지점과 2/3 지점 계산
    I = int(L // 3)
    J = 2 * I

    # 1/3 지점에서 요소 뒤집기
    if lr_state[I:I+2] == ['2', '1']:
        lr_state[I:I+2] = ['1', '2']
        if lr_state[J:J+2] == ['1', '2']:
            lr_state[J:J+2] = ['2', '1']
        else:
            lr_state[(J-1):J+1] = ['2', '1']
    else:
        lr_state[I:I+2] = ['2', '1']
        if lr_state[J:J+2] == ['2', '1']:
            lr_state[J:J+2] = ['1', '2']
        else:
            lr_state[(J-1):J+1] = ['1', '2']
    return lr_state

def measurements(psi, L):
    
    # Measurements
    Ns = psi.expectation_value("N")
    NNs = psi.expectation_value("NN")
    EE = psi.entanglement_entropy()
    
    # Measuring Correlation functions from the center
    Cnn_center = np.zeros(L)
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
            D = psi.expectation_value_term([('Bd',I),('B',I+1),('Bd',J),('B',J-1)])
            Dsp_center1[i] = D.real
            D = psi.expectation_value_term([('Bd',I),('B',I+1),('Bd',J+1),('B',J)])
            Dsp_center2[i] = D.real
        
        if i<L-2:
            Q = psi.expectation_value_term([('Bd',I),('B',I+1),('B',I+1),('Bd',I+2),('B',J+2),('Bd',J+1),('Bd',J+1),('B',J)])
            Qsp_center1[i] = Q.real
            Q = psi.expectation_value_term([('Bd',I),('B',I+1),('B',I+1),('Bd',I+2),('B',J+1),('Bd',J),('Bd',J),('B',J-1)])
            Qsp_center1[i] = Q.real

    return Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, EE


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


def write_data( Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Bcor, Ncor, Dcor, F, F_CDW, F_LR1, F_LR2, F_131, F_202, F_040, F_COH, EE, time, path ):

    ensure_dir(path+"/observables/")
    ensure_dir(path+"/mps/")

    # data = {"psi": psi}
    # with h5py.File(path+"/mps/psi_time_%.3f.h5" % time, 'w') as f:
    #     hdf5_io.save_to_hdf5(f, data)

    file_EE = open(path+"/observables/EE.txt","a", 1)    
    file_Ns = open(path+"/observables/Ns.txt","a", 1)
    file_NNs = open(path+"/observables/NNs.txt","a", 1)
    file_Cnn = open(path+"/observables/Cnn.txt","a", 1)
    file_Dsp1 = open(path+"/observables/Dsp1.txt","a", 1)
    file_Dsp2 = open(path+"/observables/Dsp2.txt","a", 1)
    file_Qsp1 = open(path+"/observables/Qsp1.txt","a", 1)
    file_Qsp2 = open(path+"/observables/Qsp2.txt","a", 1)
    
    file_EE.write(repr(time) + " " + "  ".join(map(str, EE)) + " " + "\n")
    file_Ns.write(repr(time) + " " + "  ".join(map(str, Ns)) + " " + "\n")
    file_NNs.write(repr(time) + " " + "  ".join(map(str, NNs)) + " " + "\n")
    file_Cnn.write(repr(time) + " " + "  ".join(map(str, Cnn_center)) + " " + "\n")
    file_Dsp1.write(repr(time) + " " + "  ".join(map(str, Dsp_center1)) + " " + "\n")
    file_Dsp2.write(repr(time) + " " + "  ".join(map(str, Dsp_center2)) + " " + "\n")
    file_Qsp1.write(repr(time) + " " + "  ".join(map(str, Qsp_center1)) + " " + "\n")
    file_Qsp2.write(repr(time) + " " + "  ".join(map(str, Qsp_center2)) + " " + "\n")
    
    file_EE.close()
    file_Ns.close()
    file_NNs.close()
    file_Cnn.close()
    file_Dsp1.close()
    file_Dsp2.close()
    file_Qsp1.close()
    file_Qsp2.close()
    
    #
    file = open(path+"/observables.txt","a", 1)    
    file.write(repr(time) + " " + repr(np.max(EE)) + " " + repr(EE[len(EE)//2]) + " " + repr(np.mean(Ns)) + " " + repr(np.mean(NNs)) + " " + repr(np.abs(Bcor)) + " " + repr(np.abs(Ncor)) + " " + repr(np.abs(Dcor)) + " " + repr(F) + " " + repr(F_CDW) + " " + repr(F_LR1) + " " + repr(F_LR2) + " " + repr(F_131) + " " + repr(F_202) + " " + repr(F_040) + " " + repr(F_COH) + " " + "\n")
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
    parser.add_argument("--p1", default='1.0', help="fugacity of 131 state")
    parser.add_argument("--p2", default='1.0', help="fugacity of 202 state")
    parser.add_argument("--init_state", default='2', help="Initial state")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    parser.add_argument('--autocorr', action='store_true', help='enable autocorrelation function calculation')
    parser.add_argument('--d_corr_func', action='store_true', help='enable dipole correlation function calculation')
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
    p1 = float(args.p1)
    p2 = float(args.p2)
    init_state = args.init_state
    path = args.path
    
    #################
    # before quench #
    #################
    
    model_params0 = {
    "L": L, 
    "td": td0,
    "U": U0,
    "Ncut": Ncut,
    "mu": U0
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
    'chi_list': { 0: 10}, #{ 0: 10, 5: 20, 10: chi0 },
    'max_E_err': 1.0e-9,
    'max_S_err': 1.0e-9,
    'max_sweeps': 1, #100,
    'combine' : True
    }

    # initial state
    if init_state == '1':
        product_state = ['1'] * L
    elif init_state == '2':
        product_state = ['2'] * L
    elif init_state == '1-half-a':
        if L % 2 == 0:
            product_state = ['1', '2'] * (L // 2)
            #raise ValueError("Length must be odd.")
        else:
            product_state = ['1', '2'] * (L // 2) + ['1']
    elif init_state == '1-half-b':
        if L % 4 != 2:
            raise ValueError("Length must be 4n+2.")
        else:
            product_state = ['1', '1', '2', '2'] * (L // 4) + ['1','1']
    elif init_state == 'fractional1':
        if L % 4 != 1:
            raise ValueError("Length must be 4n+1.")
        else:
            product_state = ['2', '1', '1', '1'] * (L // 4) + ['1']
    elif init_state == 'fractional2':
        if L % 4 != 1:
            raise ValueError("Length must be 4n+1.")
        else:
            product_state = ['1', '2', '2', '2'] * (L // 4) + ['1']
    elif init_state == '3':
        product_state = ['3'] * L
    elif init_state == '4':
        product_state = ['4'] * L
    elif init_state == '5':
        product_state = ['5'] * L
    
    DBHM0 = model.DIPOLAR_BOSE_HUBBARD_CONSERVED(model_params0)
    psi = MPS.from_product_state(DBHM0.lat.mps_sites(), product_state, bc=DBHM0.lat.bc_MPS)
    cdw_state = psi.copy()

    if init_state == '1-half-a':
        coh_state = coherent_state(L, Ncut, p1, p2)
        lr1_state = MPS.from_product_state(DBHM0.lat.mps_sites(), lr1_configuration(L), bc=DBHM0.lat.bc_MPS)
        lr2_state = MPS.from_product_state(DBHM0.lat.mps_sites(), lr2_configuration(L), bc=DBHM0.lat.bc_MPS)
        ex_131_state = MPS.from_product_state(DBHM0.lat.mps_sites(), ex_131_configuration(L), bc=DBHM0.lat.bc_MPS)
        ex_202_state = MPS.from_product_state(DBHM0.lat.mps_sites(), ex_202_configuration(L), bc=DBHM0.lat.bc_MPS)
        ex_040_state = MPS.from_product_state(DBHM0.lat.mps_sites(), ex_040_configuration(L), bc=DBHM0.lat.bc_MPS)
        
    eng = dmrg.TwoSiteDMRGEngine(psi, DBHM0, dmrg_params)
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
    
    Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, EE = measurements(psi, L)
    if init_state == '1-half-a':
        F_COH = np.abs(coh_state.overlap( MPS_drop_charge(psi)) )
        F_CDW = np.abs(psi.overlap(cdw_state))
        F_LR1 = np.abs(psi.overlap(lr1_state))
        F_LR2 = np.abs(psi.overlap(lr2_state))
        F_131 = np.abs(psi.overlap(ex_131_state))
        F_202 = np.abs(psi.overlap(ex_202_state))
        F_040 = np.abs(psi.overlap(ex_040_state))
        write_data( Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Bcor, Ncor, Dcor, 1.0, F_CDW, F_LR1, F_LR2, F_131, F_202, F_040, F_COH, EE, 0, path )
    else:
        write_data( Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Bcor, Ncor, Dcor, 1.0, 0., 0., 0., 0., 0., 0., 0., EE, 0, path )

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
            Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, EE = measurements(psi, L)

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

            F = np.abs(psi.overlap(psi0))
            if init_state == '1-half-a':
                F_COH = np.abs(coh_state.overlap( MPS_drop_charge(psi,permute_p_leg=False)) )
                F_CDW = np.abs(psi.overlap(cdw_state))
                F_LR1 = np.abs(psi.overlap(lr1_state))
                F_LR2 = np.abs(psi.overlap(lr2_state))
                F_131 = np.abs(psi.overlap(ex_131_state))
                F_202 = np.abs(psi.overlap(ex_202_state))
                F_040 = np.abs(psi.overlap(ex_040_state))
                write_data( Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Bcor, Ncor, Dcor, F, F_CDW, F_LR1, F_LR2, F_131, F_202, F_040, F_COH, EE, tdvp_engine.evolved_time, path )
            else:
                write_data( Ns, NNs, Cnn_center, Dsp_center1, Dsp_center2, Qsp_center1, Qsp_center2, Bcor, Ncor, Dcor, F, 0., 0., 0., 0., 0., 0., 0., EE, tdvp_engine.evolved_time, path )


            if args.d_corr_func:
                dc_corr_func(psi, L, tdvp_engine.evolved_time, path)

        if (i+1) % Pstep == 0:
            
            data = {"psi": psi}
            with h5py.File(path+"/mps/psi_time_%.3f.h5" % tdvp_engine.evolved_time, 'w') as f:
                hdf5_io.save_to_hdf5(f, data)