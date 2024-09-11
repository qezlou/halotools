r"""
This module contains occupation components used by the environmentally dependent model of Hadzhiyska et al 2023.
"""

import numpy as np
from scipy.special import erf
from scipy.stats import norm
import warnings

from .occupation_model_template import OccupationComponent
from .zheng07_components import Zheng07Cens, Zheng07Sats
from .. import model_defaults

class Hadzhiyska23Cens(OccupationComponent):
    """
    Central occupation model for the environmentally dependent model of Hadzhiyska et al 2023.
    Equation 13 in Hadzhiyska et al 2020 (https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.2507H/abstract)
    """

    def __init__(self,
                 threshold = model_defaults.default_luminosity_threshold,
                 prim_haloprop_key = model_defaults.prim_haloprop_key,
                 cen_no_assembias_model = None,
                 **kwargs):
        """
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. If specified,
            input value must agree with one of the thresholds used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.
        prim_haloprop_key : str
            String giving the column name of the primary halo property governing the occupation statistics.
        env_keys : list of strings
            List of strings giving the column names of the environmental properties governing the occupation statistics.
        cen_no_assembias_model : object, optional
            A model for the mean number of centrals in a halo considering only mass dependence.
            It is used to introduce the assembly bias in the mean occupation as prescribed in Eq 13 
            of Hadzhiyska et al 2020. If not passed, the default is Zheng07Cens.
        kwargs : dict
            keywords passed to the model
        Examples
        --------
        >>> cenocc = HMQCens(prim_haloprop_key='halo_mvir')
        """
        # Importing this module nags the user about an "H" attribute, let's put a placeholder as a workaround
        upper_occupation_bound = 1.0
        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Hadzhiyska23Cens, self).__init__(
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

        if cen_no_assembias_model is None:
            self.cen_no_assembias_model = Zheng07Cens(threshold=threshold,
                                                      prim_haloprop_key=prim_haloprop_key,
                                                      **kwargs)

    def mean_occupation(self, **kwargs):
        """
        Compute the mean occupation of centrals.
        See Eq 13 of Hadzhiyska et al 2020 [https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.2507H/abstract]

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
            env_term = np.zeros((0, prim_haloprop.size))
            print(f'tabl.keys(): {kwargs["table"].keys()}')
            for i, env_key in enumerate(self.param_dict['env-props']):
                env_term += self.param_dict['env-scales'][i]*kwargs['table'][env_key]
        else:
            print(f'No Table was passed!')      
        # Get the mass-dependant mean occupation of choice (e.g. Zheng07Model or HmqModel)
        # And introduce the assembly bias in the mean occupation as prescribed in Eq 13 
        # of Hadzhiyska et al 2020
        mean_ncen = self.cen_no_assembias_model.mean_occupation()       
        mean_ncen = (1 + env_term * ( 1 - mean_ncen)) * mean_ncen

        return mean_ncen
    
    def _initialize_param_dict(self):
        """
        Set the initial values of ``self.param_dict`` from Table 1 of Alam et al 2020.
        """
        self.param_dict = {"env-props":["Shear-s1.0"], 
                           "env-scales":[1]}

class Hadzhiyska23Sats(OccupationComponent):
    """
    Central occupation model for the environmentally dependent model of Hadzhiyska et al 2023.
    Equation 13 in Hadzhiyska et al 2020 (https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.2507H/abstract)
    """

    def __init__(self, 
                 threshold = model_defaults.default_luminosity_threshold,
                 prim_haloprop_key = model_defaults.prim_haloprop_key,
                 sat_no_assembias_model = None,
                 env_keys = ['Shear-s1.0'],
                 **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. If specified,
            input value must agree with one of the thresholds used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.
        prim_haloprop_key : str
            String giving the column name of the primary halo property governing the occupation statistics.
        env_keys : list of strings
            List of strings giving the column names of the environmental properties governing the occupation statistics.
        sat_no_assembias_model : object, optional
            A model for the mean number of satellites in a halo considering only mass dependence.
            It is used to introduce the assembly bias in the mean occupation as prescribed in Eq 13 
            of Hadzhiyska et al 2020. If not passed, the default is Zheng07Cens.
        kwargs : dict
            keywords passed to the model
        Examples
        --------
        >>> cenocc = HMQCens(prim_haloprop_key='halo_mvir')
        """
        # Importing this module nags the user about an "H" attribute, let's put a placeholder as a workaround
        upper_occupation_bound = 1.0
        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Hadzhiyska23Sats, self).__init__(
            gal_type="satellites",
            threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)
        self._methods_to_inherit = ([
            "mc_occupation",
            "mean_occupation"
        ])

        self._initialize_param_dict()
        self.param_dict.update(kwargs)


        if sat_no_assembias_model is None:
            self.sat_no_assembias_model = Zheng07Sats(threshold=threshold,
                                                      prim_haloprop_key=prim_haloprop_key,
                                                      **kwargs)
        self.env_keys = env_keys

    def mean_occupation(self, **kwargs):
        """
        Compute the mean occupation of centrals.
        See Eq 13 of Hadzhiyska et al 2020 [https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.2507H/abstract]

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
            env_term = np.zeros((0, prim_haloprop.size))
            for i, env_key in enumerate(self.param_dict['env-props']):
                env_term += self.param_dict['env-scales'][i]*kwargs['table'][env_key]
        # Get the mass-dependant mean occupation of choice (e.g. Zheng07Model or HmqModel)
        # And introduce the assembly bias in the mean occupation as prescribed in Eq 13 
        # of Hadzhiyska et al 2020
        mean_nsat = self.sat_no_assembias_model.mean_occupation()
        
        mean_nsat = (1 + env_term * ( 1 - mean_nsat)) * mean_nsat
        
        return mean_nsat
    
    def _initialize_param_dict(self):
        """
        Set the initial values of ``self.param_dict`` from Table 1 of Alam et al 2020.
        """
        self.param_dict = {"env-props":["Shear-s1.0"], 
                           "env-scales":[1]}
