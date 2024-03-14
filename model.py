# Copyright 2023 Hyun-Yong Lee

from tenpy.models.lattice import Chain, Lattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.tools.params import Config
from tenpy.networks.site import BosonSite, SpinHalfSite, GroupedSite
import numpy as np
import sym_sites
__all__ = ['DIPOLAR_BOSE_HUBBARD']


class DIPOLAR_BOSE_HUBBARD(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "DIPOLAR_BOSE_HUBBARD")
        L = model_params.get('L', 1)
        td = model_params.get('td', 1.)
        ts = model_params.get('ts', 0.)
        U = model_params.get('U', 0.)
        Ncut = model_params.get('Ncut', 2)
        
        site = BosonSite( Nmax=Ncut, conserve='N', filling=0.0 )
        site.multiply_operators(['B','B'])
        site.multiply_operators(['Bd','Bd'])

        lat = Chain( L=L, site=site, bc='open', bc_MPS='finite', order='default' )
        CouplingModel.__init__(self, lat)

        # 2-site hopping
        self.add_coupling( -ts, 0, 'B', 0, 'Bd', 1, plus_hc=True)
        
        # 3-site hopping
        self.add_multi_coupling( -td, [('Bd', 0, 0), ('B B', 1, 0), ('Bd', 2, 0)])
        self.add_multi_coupling( -td, [('B', 0, 0), ('Bd Bd', 1, 0), ('B', 2, 0)])

        # 4-site hopping
        # self.add_multi_coupling( -td, [('Bd', 0, 0), ('B', 1, 0), ('B', 2, 0), ('Bd', 3, 0)])
        # self.add_multi_coupling( -td, [('B', 0, 0), ('Bd', 1, 0), ('Bd', 2, 0), ('B', 3, 0)])

        # Onsite Hubbard Interaction
        self.add_onsite( U/2., 0, 'NN')

        MPOModel.__init__(self, lat, self.calc_H_MPO())



class DIPOLAR_BOSE_HUBBARD_CONSERVED(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "DIPOLAR_BOSE_HUBBARD")
        L = model_params.get('L', 1)
        td = model_params.get('td', 1.)
        U = model_params.get('U', 0.)
        mu = model_params.get('mu', 0.)
        Ncut = model_params.get('Ncut', 2)
        
        sites = [ sym_sites.BosonSite_DM_conserved(Nmax=Ncut, cons_N='N', cons_D='D', x=x) for x in range(L) ]
        # sites = [ sym_sites.BosonSite_DM_conserved(Nmax=Ncut, cons_N='N', cons_D=None, x=x) for x in range(L) ]
        
        for x in range(L):
            sites[x].multiply_operators(['B','B'])
            sites[x].multiply_operators(['Bd','Bd'])

        lat = Lattice([1], sites, order='default', bc='open', bc_MPS='finite', basis=[[L]], positions=np.array([range(L)]).transpose())
        
        CouplingModel.__init__(self, lat)

        # 3-site hopping    
        for x in range(L-2):
            self.add_multi_coupling( -td, [('Bd', 0, x), ('B B', 0, x+1), ('Bd', 0, x+2)])
            self.add_multi_coupling( -td, [('B', 0, x), ('Bd Bd', 0, x+1), ('B', 0, x+2)])

        # onsite Hubbard Interaction
        for x in range(L):
            self.add_onsite( U/2., x, 'NN')
            self.add_onsite( -U/2., x, 'N')
            self.add_onsite( -mu, x, 'N')

        MPOModel.__init__(self, lat, self.calc_H_MPO())


class EFFECTIVE_PXP(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "DIPOLAR_BOSE_HUBBARD")
        L = model_params.get('L', 1)
        J = model_params.get('J', 1.)
        U = model_params.get('U', 1.)

        site = SpinHalfSite(conserve=None)
        site_L = GroupedSite([site, site])
        
        P = np.array([[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
        P1 = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
        P2 = np.array([[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.]])
        X1 = np.array([[0.,0.,1.,0.],[0.,0.,0.,1.],[1.,0.,0.,0.],[0.,1.,0.,0.]])
        X2 = np.array([[0.,1.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]])
        N1 = np.array([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
        N2 = np.array([[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.]])
        
        site_L.add_op('P', P)
        site_L.add_op('P1', P1)
        site_L.add_op('P2', P2)
        site_L.add_op('X1', X1)
        site_L.add_op('X2', X2)
        site_L.add_op('N1', N1)
        site_L.add_op('N2', N2)
        
        lat = Chain( L=L, site=site_L, bc='open', bc_MPS='finite', order='default' )
        CouplingModel.__init__(self, lat)

        # PXP    
        # self.add_multi_coupling( -np.sqrt(6)*J, [('P', 0, 0), ('X1', 1, 0), ('P', 2, 0)])
        # self.add_multi_coupling( -np.sqrt(2)*J, [('P', 0, 0), ('X2', 1, 0), ('P', 2, 0)])
        
        self.add_multi_coupling( -np.sqrt(6)*J, [('P2', 0, 0), ('X1', 1, 0), ('P2', 1, 0)])
        self.add_multi_coupling( -np.sqrt(2)*J, [('P2', 0, 0), ('P1', 1, 0), ('X2', 1, 0), ('P', 2, 0)])
        
        # Onsite Hubbard Interaction
        self.add_onsite( U, 0, 'N1')
        self.add_onsite( U, 0, 'N2')

        MPOModel.__init__(self, lat, self.calc_H_MPO())


class EFFECTIVE_PXP2(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "DIPOLAR_BOSE_HUBBARD")
        L = model_params.get('L', 1)
        J = model_params.get('J', 1.)
        U = model_params.get('U', 1.)

        site = SpinHalfSite(conserve=None)
        P = np.array([[1.,0.],[0.,0.]])
        N = np.array([[0,0.],[0.,1]])
        site.add_op('P', P)
        site.add_op('N', N)
        
        basis = [ [2,0], [0,1] ]
        pos = [ [0,0], [1,0] ]
        nn = [ (0, 1, [0,0]), (1, 0, [1,0]) ] 
        lat = Lattice( Ls=[L, 1], unit_cell=[site, site], basis=basis, positions=pos, bc='open', bc_MPS='finite', nearest_neighbors=nn)
        CouplingModel.__init__(self, lat)

        # PXP    
        # self.add_multi_coupling( -np.sqrt(6)*J, [('P', 0, 0), ('X1', 1, 0), ('P', 2, 0)])
        # self.add_multi_coupling( -np.sqrt(2)*J, [('P', 0, 0), ('X2', 1, 0), ('P', 2, 0)])
        
        self.add_multi_coupling( -2*np.sqrt(6)*J, [('P',[-1,0],1),('Sigmax',[0,0],0),('P',[0,0],1)] )
        self.add_multi_coupling( -2*np.sqrt(2)*J, [('P',[-1,0],0),('P',[-1,0],1),('Sigmax',[0,0],1),('P',[1,0],0),('P',[1,0],1)] )
        
        # Onsite Hubbard Interaction
        self.add_onsite( U, 0, 'N')
        self.add_onsite( U, 1, 'N')

        MPOModel.__init__(self, lat, self.calc_H_MPO())

