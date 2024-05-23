r"""
This module contains occupation components used by the Hmq (high-mass quenched) composite model.
"""

import numpy as np
from scipy.special import erf
from scipy.stats import norm
import warnings

from .occupation_model_template import OccupationComponent
from .. import model_defaults, model_helpers
from ..smhm_models import Behroozi10SmHm

from ...utils.array_utils import custom_len
from ... import sim_manager
from ...custom_exceptions import HalotoolsError
#__all__ = ('Alam20Cens')


class Alam20Cens(OccupationComponent):
    """
    Central occupation model for the HMQ model.
    Alam et al 2009. Equations in Hadzhiyska et al 2020 (https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.2507H/abstract)
    are clearer, so we will use those.
    """

    def __init__(self,
                 threshold=model_defaults.default_luminosity_threshold,
                 prim_haloprop_key=model_defaults.prim_haloprop_key,
                 **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : str
            String giving the column name of the primary halo property governing the occupation statistics.
        kwargs : dict
            keywords passed to the model
        Examples
        --------
        >>> cenocc = Alam20Cens(prim_haloprop_key='halo_mvir')
        """
        # Importing this module nags the user about an "H" attribute, let's put a placeholder as a workaround
        upper_occupation_bound = 1.0
        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Alam20Cens, self).__init__(
            gal_type="centrals",
            threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs
        )
        self._methods_to_inherit = ([
            "mc_occupation",
            "mean_occupation"
        ])
        self._initialize_param_dict()
        self.param_dict.update(kwargs)

    def mean_occupation(self, **kwargs):
        """
        Compute the mean occupation of centrals.
        See Eq 3-6 of Hadzhiyska et al 2023 [https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.2507H/abstract]

        Parameters
        ----------
        prim_haloprop : array, optional
            array of mass-like variable upon which occupation statistics are based.
            if ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.
        table : object, optional
            Data table storing halo catalog.
            if ``table`` is not passed, then ``prim_haloprop`` must be passed.
        kwargs : dict
            keywords passed to the model

        Returns
        -------
        mean_ncen : array
            mean number of centrals in the halo of the input mass
        """
        if 'table' in kwargs:
            prim_haloprop = kwargs['table'][self.prim_haloprop_key]
        
        gaussian_part = norm(loc= np.log10(self.param_dict['m_cut']), 
                   scale= self.param_dict['sigma_logm']).pdf(np.log10(prim_haloprop))
        
        fm = (np.log10(prim_haloprop) - np.log10(self.param_dict['m_cut'])) / self.param_dict['sigma_logm']
        erf_part = 0.5 * (1 + erf(self.param_dict['gamma']*fm / np.sqrt(2)))
        amp = (self.param_dict['p_max'] - 1/self.param_dict['q']) / np.max(2* gaussian_part * erf_part)
        add_part = (1/(2*self.param_dict['q']) * 
                    (1 + erf((np.log10(prim_haloprop) - np.log10(self.param_dict['m_cut']))/0.01)))
        
        return 2*amp * gaussian_part * erf_part + add_part
    
    def _initialize_param_dict(self):
        """
        Set the initial values of ``self.param_dict`` from Table 1 of Alam et al 2020.
        """
        self.param_dict = {"m_cut": 10**11.75}
        self.param_dict["sigma_logm"] = 0.58
        self.param_dict["gamma"] = 4.12
        self.param_dict["q"] = 100
        self.param_dict["log10(M_1)"] = 13.53
        self.param_dict["p_max"] = 0.33


