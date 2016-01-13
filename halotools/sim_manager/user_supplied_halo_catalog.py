""" Module containing the UserSuppliedHaloCatalog class. 
"""

import numpy as np
import os, sys, urllib2, fnmatch
from warnings import warn 
import datetime 

from astropy import cosmology
from astropy import units as u
from astropy.table import Table

try:
    import h5py
except ImportError:
    warn("Most of the functionality of the sim_manager "
        "sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda")

from .halo_table_cache import HaloTableCache 
from .halo_table_cache_log_entry import HaloTableCacheLogEntry, get_redshift_string

from ..utils.array_utils import custom_len, convert_to_ndarray
from ..custom_exceptions import HalotoolsError

__all__ = ('UserSuppliedHaloCatalog', )

class UserSuppliedHaloCatalog(object):
    """ Class used to transform a user-provided halo catalog 
    into the standard form recognized by Halotools. 

    See :ref:`user_supplied_halo_catalogs` for a tutorial on this class. 
    
    """
    def __init__(self, **kwargs):
        """
        Parameters 
        ------------
        **metadata : float or string 
            Keyword arguments storing catalog metadata. 
            The quantities `Lbox` and `particle_mass` 
            are required and must be in Mpc/h and Msun/h units, respectively. 
            `redshift` is also required metadata. 
            See Examples section for further notes. 

        **halo_catalog_columns : sequence of arrays 
            Sequence of length-*Nhalos* arrays passed in as keyword arguments. 

            Each key will be the column name attached to the input array. 
            All keys must begin with the substring ``halo_`` to help differentiate 
            halo property from mock galaxy properties. At a minimum, there must be a 
            ``halo_id`` keyword argument storing a unique integer for each halo, 
            as well as columns ``halo_x``, ``halo_y`` and ``halo_z``. 
            There must also be some additional mass-like variable, 
            for which you can use any name that begins with ``halo_``
            See Examples section for further notes. 

        ptcl_table : table, optional 
            Astropy `~astropy.table.Table` object storing dark matter particles 
            randomly selected from the snapshot. At a minimum, the table must have 
            columns ``x``, ``y`` and ``z``. 

        Notes 
        -------
        This class is tested by 
        `~halotools.sim_manager.tests.test_user_supplied_halo_catalog.TestUserSuppliedHaloCatalog`. 

        Examples 
        ----------
        Here is an example using dummy data to show how to create a new `UserSuppliedHaloCatalog` 
        instance from from your own halo catalog. First the setup:

        >>> redshift = 0.0
        >>> Lbox = 250.
        >>> particle_mass = 1e9
        >>> num_halos = 100
        >>> x = np.random.uniform(0, Lbox, num_halos)
        >>> y = np.random.uniform(0, Lbox, num_halos)
        >>> z = np.random.uniform(0, Lbox, num_halos)
        >>> mass = np.random.uniform(1e12, 1e15, num_halos)
        >>> ids = np.arange(0, num_halos)

        Now we simply pass in both the metadata and the halo catalog columns as keyword arguments:

        >>> halo_catalog = UserSuppliedHaloCatalog(redshift = redshift, Lbox = Lbox, particle_mass = particle_mass, halo_x = x, halo_y = y, halo_z = z, halo_id = ids, halo_mvir = mass)

        Your ``halo_catalog`` object can be used throughout the Halotools package. 
        The halo catalog itself is stored in the ``halo_table`` attribute, with columns accessed as follows:

        >>> array_of_masses = halo_catalog.halo_table['halo_mvir']
        >>> array_of_x_positions = halo_catalog.halo_table['halo_x']

        Each piece of metadata you passed in can be accessed as an ordinary attribute:

        >>> halo_catalog_box_size = halo_catalog.Lbox
        >>> particle_mass = halo_catalog.particle_mass

        If you wish to pass in additional metadata, just include additional keywords:

        >>> simname = 'my_personal_sim'

        >>> halo_catalog = UserSuppliedHaloCatalog(redshift = redshift, simname = simname, Lbox = Lbox, particle_mass = particle_mass, halo_x = x, halo_y = y, halo_z = z, halo_id = ids, halo_mvir = mass)

        Similarly, if you wish to include additional columns for your halo catalog, 
        Halotools is able to tell the difference between metadata and columns of halo data:

        >>> spin = np.random.uniform(0, 0.2, num_halos)
        >>> halo_catalog = UserSuppliedHaloCatalog(redshift = redshift, halo_spin = spin, simname = simname, Lbox = Lbox, particle_mass = particle_mass, halo_x = x, halo_y = y, halo_z = z, halo_id = ids, halo_mvir = mass)

        If you want to store your halo catalog in the Halotools cache, 
        use the `add_halocat_to_cache` method. 

        """
        halo_table_dict, metadata_dict = self._parse_constructor_kwargs(**kwargs)
        self.halo_table = Table(halo_table_dict)

        self._test_metadata_dict(**metadata_dict)
        for key, value in metadata_dict.iteritems():
            setattr(self, key, value)

        self._passively_bind_ptcl_table(**kwargs)

    def _parse_constructor_kwargs(self, **kwargs):
        """ Private method interprets constructor keyword arguments and returns two 
        dictionaries. One stores the halo catalog columns, the other stores the metadata. 

        Parameters 
        ------------
        **kwargs : keyword arguments passed to constructor 

        Returns 
        ----------
        halo_table_dict : dictionary 
            Keys are the names of the halo catalog columns, values are length-*Nhalos* ndarrays. 

        metadata_dict : dictionary 
            Dictionary storing the catalog metadata. Keys will be attribute names bound 
            to the `UserSuppliedHaloCatalog` instance. 
        """

        try:
            halo_id = kwargs['halo_id']
            assert type(halo_id) is np.ndarray 
            Nhalos = custom_len(halo_id)
            assert Nhalos > 1
        except KeyError, AssertionError:
            msg = ("\nThe UserSuppliedHaloCatalog requires a ``halo_id`` keyword argument "
                "storing an ndarray of length Nhalos > 1.\n")
            raise HalotoolsError(msg)

        halo_table_dict = (
            {key: kwargs[key] for key in kwargs 
            if (type(kwargs[key]) is np.ndarray) 
            and (custom_len(kwargs[key]) == Nhalos) 
            and (key[:5] == 'halo_')}
            )
        self._test_halo_table_dict(halo_table_dict)

        metadata_dict = (
            {key: kwargs[key] for key in kwargs
            if (key not in halo_table_dict) and (key != 'ptcl_table')}
            )

        return halo_table_dict, metadata_dict 


    def _test_halo_table_dict(self, halo_table_dict):
        """
        """ 
        try:
            assert 'halo_x' in halo_table_dict 
            assert 'halo_y' in halo_table_dict 
            assert 'halo_z' in halo_table_dict 
            assert len(halo_table_dict) >= 5
        except AssertionError:
            msg = ("\nThe UserSuppliedHaloCatalog requires keyword arguments ``halo_x``, "
                "``halo_y`` and ``halo_z``,\nplus one additional column storing a mass-like variable.\n"
                "Each of these keyword arguments must storing an ndarray of the same length\n"
                "as the ndarray bound to the ``halo_id`` keyword argument.\n")
            raise HalotoolsError(msg)

    def _test_metadata_dict(self, **metadata_dict):
        """
        """
        try:
            assert 'Lbox' in metadata_dict
            assert custom_len(metadata_dict['Lbox']) == 1
            assert 'particle_mass' in metadata_dict
            assert custom_len(metadata_dict['particle_mass']) == 1
            assert 'redshift' in metadata_dict
        except AssertionError:
            msg = ("\nThe UserSuppliedHaloCatalog requires "
                "keyword arguments ``Lbox``, ``particle_mass`` and ``redshift``\n"
                "storing scalars that will be interpreted as metadata about the halo catalog.\n")
            raise HalotoolsError(msg)

        Lbox = metadata_dict['Lbox']
        try:
            x, y, z = (
                self.halo_table['halo_x'], 
                self.halo_table['halo_y'], 
                self.halo_table['halo_z']
                )
            assert np.all(x >= 0)
            assert np.all(x <= Lbox)
            assert np.all(y >= 0)
            assert np.all(y <= Lbox)
            assert np.all(z >= 0)
            assert np.all(z <= Lbox)
        except AssertionError:
            msg = ("The ``halo_x``, ``halo_y`` and ``halo_z`` columns must only store arrays\n"
                "that are bound by 0 and the input ``Lbox``. \n")
            raise HalotoolsError(msg)

        redshift = metadata_dict['redshift']
        try:
            assert type(redshift) == float
        except AssertionError:
            msg = ("\nThe ``redshift`` metadata must be a float.\n")
            raise HalotoolsError(msg)

        for key, value in metadata_dict.iteritems():
            if (type(value) == np.ndarray):
                if custom_len(value) == len(self.halo_table['halo_id']):
                    msg = ("\nThe input ``" + key + "`` argument stores a length-Nhalos ndarray.\n"
                        "However, this key is being interpreted as metadata because \n"
                        "it does not begin with ``halo_``. If this is your intention, ignore this message.\n"
                        "Otherwise, rename this key to begin with ``halo_``. \n")
                    warn(msg, UserWarning)


    def _passively_bind_ptcl_table(self, **kwargs):
        """
        """

        try:
            ptcl_table = kwargs['ptcl_table']

            assert type(ptcl_table) is Table
            assert len(ptcl_table) >= 1e4
            assert 'x' in ptcl_table.keys()
            assert 'y' in ptcl_table.keys()
            assert 'z' in ptcl_table.keys()

            self.ptcl_table = ptcl_table

        except AssertionError:
            msg = ("\nIf passing a ``ptcl_table`` to UserSuppliedHaloCatalog, \n"
                "this argument must contain an Astropy Table object with at least 1e4 rows\n"
                "and ``x``, ``y`` and ``z`` columns. \n")
            raise HalotoolsError(msg)

        except KeyError:
            pass


    def add_halocat_to_cache(self, 
        fname, simname, halo_finder, version_name, processing_notes, 
        overwrite = False, **additional_metadata):
        """
        Parameters 
        ------------
        fname : string 
            Absolute path of the file to be stored in cache. 
            Must conclude with an `.hdf5` extension. 

        simname : string 
            Nickname of the simulation used as a shorthand way to keep track 
            of the halo catalogs in your cache. 

        halo_finder : string 
            Nickname of the halo-finder used to generate the hlist file from particle data. 

        version_name : string 
            Nickname of the version of the halo catalog. 
            The ``version_name`` is used as a bookkeeping tool in the cache log.

        processing_notes : string, optional 
            String used to provide supplementary notes that will be attached to 
            the hdf5 file storing your halo catalog. 

        overwrite : bool, optional 
            If the chosen ``fname`` already exists, then you must set ``overwrite`` 
            to True in order to write the file to disk. Default is False. 

        **additional_metadata : sequence of strings, optional 
            Each keyword of ``additional_metadata`` defines the name 
            of a piece of metadata stored in the hdf5 file. The 
            value bound to each key can be any string. When you load your 
            cached halo catalog into memory, each piece of metadata 
            will be stored as an attribute of the 
            `~halotools.sim_manager.CachedHaloCatalog` instance. 

        """
        try:
            import h5py 
        except ImportError:
            msg = ("\nYou must have h5py installed if you want to \n"
                "store your catalog in the Halotools cache. \n")
            raise HalotoolsError(msg)

        ############################################################
        ## Perform some consistency checks in the fname
        if (os.path.isfile(fname)) & (overwrite == False):
            msg = ("\nYou attempted to store your halo catalog "
                "in the following location: \n\n" + str(fname) + 
                "\n\nThis path points to an existing file. \n"
                "Either choose a different fname or set ``overwrite`` to True.\n")
            raise HalotoolsError(msg)

        try:
            dirname = os.path.dirname(fname)
            assert os.path.exists(dirname)
        except:
            msg = ("\nThe directory you are trying to store the file does not exist. \n")
            raise HalotoolsError(msg)

        if fname[-5:] != '.hdf5':
            msg = ("\nThe fname must end with an ``.hdf5`` extension.\n")
            raise HalotoolsError(msg)
        ############################################################
        ## Perform consistency checks on the remaining log entry attributes
        try:
            _ = str(simname)
            _ = str(halo_finder)
            _ = str(version_name)
            _ = str(processing_notes)
        except:
            msg = ("\nThe input ``simname``, ``halo_finder``, ``version_name`` "
                "and ``processing_notes``\nmust all be strings.")
            raise HalotoolsError(msg)

        for key, value in additional_metadata.iteritems():
            try:
                _ = str(value)
            except:
                msg = ("\nIf you use ``additional_metadata`` keyword arguments \n"
                    "to provide supplementary metadata about your catalog, \n"
                    "all such metadata will be bound to the hdf5 file in the "
                    "format of a string.\nHowever, the value you bound to the "
                    "``"+key+"`` keyword is not representable as a string.\n")
                raise HalotoolsError(msg)


        ############################################################
        ## Now write the file to disk and add the appropriate metadata 

        self.halo_table.write(fname, path='data', overwrite = overwrite)

        f = h5py.File(fname)

        redshift_string = str(get_redshift_string(self.redshift))

        f.attrs.create('simname', str(simname))
        f.attrs.create('halo_finder', str(halo_finder))
        f.attrs.create('version_name', str(version_name))
        f.attrs.create('redshift', redshift_string)
        f.attrs.create('fname', str(fname))

        f.attrs.create('Lbox', self.Lbox)
        f.attrs.create('particle_mass', self.particle_mass)

        time_right_now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.attrs.create('time_catalog_was_originally_cached', time_right_now)

        f.attrs.create('processing_notes', str(processing_notes))

        for key, value in additional_metadata.iteritems():
            f.attrs.create(key, str(value))

        f.close()
        ############################################################
        # Now that the file is on disk, add it to the cache
        cache = HaloTableCache()

        log_entry = HaloTableCacheLogEntry(simname = simname, 
            halo_finder = halo_finder, version_name = version_name, 
            redshift = self.redshift, fname = fname)

        cache.add_entry_to_cache_log(log_entry, update_ascii = True)
        self.log_entry = log_entry

    def add_ptcl_table_to_cache(self):
        """ Not implemented yet. 
        """
        raise NotImplementedError













