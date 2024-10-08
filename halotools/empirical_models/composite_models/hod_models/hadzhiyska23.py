r"""
Module containing the HOD-style composite model
published in Zheng et al. (2007), arXiv:0703457.
"""

import numpy as np
import h5py

from ... import model_defaults
from ...occupation_models import hadzhiyska23_components
from ...phase_space_models import NFWPhaseSpace, TrivialPhaseSpace

from ....sim_manager import sim_defaults
from ....custom_exceptions import HalotoolsError

__all__ = ['hadzhiyska23_model_dictionary']

class EnvProp(object):
    def __init__(self, prop_key, prop_file, gal_type):
        self.gal_type = gal_type
        self.prop_file = prop_file
        self.prop_key = prop_key
        self._mock_generation_calling_sequence = ['assign_env_prop']
        print(f'prop_key: {prop_key}')
        self._galprop_dtypes_to_allocate = np.dtype([(prop_key, 'f4')])
    
    def assign_env_prop(self, **kwargs):
        table = kwargs['table']
        round_x = np.floor(table['halo_x']).astype(int)
        round_y = np.floor(table['halo_y']).astype(int)
        round_z = np.floor(table['halo_z']).astype(int)
        with h5py.File(self.prop_file, 'r') as f:
            table[self.prop_key] = f[self.prop_key][:][round_x,
                                                       round_y,
                                                       round_z]


def hadzhiyska23_model_dictionary(
        threshold=model_defaults.default_luminosity_threshold,
        redshift=sim_defaults.default_redshift, modulate_with_cenocc=False, 
        **kwargs):
    r""" Dictionary for an HOD-style based on Zheng et al. (2007), arXiv:0703457.

    See :ref:`hadzhiyska23_composite_model` for a tutorial on this model.

    There are two populations, centrals and satellites.
    Central occupation statistics are given by a nearest integer distribution
    with first moment given by an ``erf`` function; the class governing this
    behavior is `~halotools.empirical_models.Hadzhiyska23Cens`.
    Central galaxies are assumed to reside at the exact center of the host halo;
    the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`.

    Satellite occupation statistics are given by a Poisson distribution
    with first moment given by a power law that has been truncated at the low-mass end;
    the class governing this behavior is `~halotools.empirical_models.Hadzhiyska23Sats`;
    satellites in this model follow an (unbiased) NFW profile, as governed by the
    `~halotools.empirical_models.NFWPhaseSpace` class.

    This composite model is built by the `~halotools.empirical_models.HodModelFactory`.

    Parameters
    ----------
    threshold : float, optional
        Luminosity threshold of the galaxy sample being modeled.
        Default is set in the `~halotools.empirical_models.model_defaults` module.

    redshift : float, optional
        Redshift of the galaxy population being modeled.
        If you will be using the model instance to populate mock catalogs,
        you must choose a redshift that is consistent with the halo catalog.
        Default is set in the `~halotools.empirical_models.model_defaults` module.

    modulate_with_cenocc : bool, optional
        If set to True, the `Hadzhiyska23Sats.mean_occupation` method will
        be multiplied by the the first moment of the centrals:

        :math:`\langle N_{\mathrm{sat}}\rangle_{M}\Rightarrow\langle N_{\mathrm{sat}}\rangle_{M}\times\langle N_{\mathrm{cen}}\rangle_{M}`

        The :math:`\langle N_{\mathrm{cen}}\rangle_{M}` function is calculated
        according to `Hadzhiyska23Cens.mean_occupation`.

    Returns
    -------
    model_dictionary : dict
        Dictionary of keywords to be passed to
        `~halotools.empirical_models.HodModelFactory`

    Examples
    --------
    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> model_instance = PrebuiltHodModelFactory('Hadzhiyska23', threshold = -21)

    As with all instances of the `~halotools.empirical_models.PrebuiltHodModelFactory`,
    you can populate a mock by passing the model a halo catalog:

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim(redshift = model_instance.redshift)
    >>> model_instance.populate_mock(halocat)

    Notes
    ------
    Although the ``cenocc_model`` is a legitimate keyword for the
    `~halotools.empirical_models.Hadzhiyska23Sats` class, this keyword is not permissible
    when building the ``hadzhiyska23`` composite model with
    the `~halotools.empirical_models.PrebuiltHodModelFactory`.
    To build a composite model that uses this feature,
    you will need to use the `~halotools.empirical_models.HodModelFactory`
    directly. See :ref:`hadzhiyska23_using_cenocc_model_tutorial` for explicit instructions.

    """
    ####################################
    # Load the environmental properties of interest form an external
    # hdf5 file and assign them to the mock
    prop_key = kwargs.pop('prop_key')
    prop_file = kwargs.pop('prop_file')
    cen_env_prop = EnvProp(prop_key=prop_key, prop_file=prop_file, gal_type='centrals')
    sat_env_prop = EnvProp(prop_key=prop_key, prop_file=prop_file, gal_type='satellites')

    ####################################
    # Build the `occupation` feature
    centrals_occupation = hadzhiyska23_components.Hadzhiyska23Cens(
        threshold=threshold, redshift=redshift, **kwargs)

    # Build the `profile` feature
    centrals_profile = TrivialPhaseSpace(redshift=redshift, **kwargs)

    ####################################
    # Build the occupation model
    cenocc_model = centrals_occupation if modulate_with_cenocc else None

    if 'cenocc_model' in list(kwargs.keys()):
        msg = ("Do not pass in the ``cenocc_model`` keyword to ``hadzhiyska23_model_dictionary``.\n"
            "The model bound to this keyword will be automatically chosen to be Hadzhiyska23Cens \n")
        raise HalotoolsError(msg)

    satellites_occupation = hadzhiyska23_components.Hadzhiyska23Sats(
        threshold=threshold, redshift=redshift,
        cenocc_model=cenocc_model, modulate_with_cenocc=modulate_with_cenocc, **kwargs)
    satellites_occupation._suppress_repeated_param_warning = True

    # Build the profile model
    satellites_profile = NFWPhaseSpace(redshift=redshift, **kwargs)

    return (({'centrals_env_prop': cen_env_prop,
             'centrals_occupation': centrals_occupation,
             'centrals_profile': centrals_profile,
             'satellites_env_prop': sat_env_prop,
             'satellites_occupation': satellites_occupation,
             'satellites_profile': satellites_profile}),(
             {'model_feature_calling_sequence':('centrals_env_prop', 
                                               'centrals_occupation', 
                                               'centrals_profile', 
                                               'satellites_env_prop',
                                               'satellites_occupation', 
                                               'satellites_profile')}))
