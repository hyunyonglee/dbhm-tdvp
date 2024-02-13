# Copyright 2023 Hyun-Yong Lee

from tenpy.models.lattice import Chain, Lattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.tools.params import Config
from tenpy.networks.site import BosonSite
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
