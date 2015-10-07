"""
grid.core.weight
================

A distance-dependent radar data objective analysis weight class used for
defining the type of weight to use for mapping (interpolating) volumetric radar
data.

A key reference for radar data objective analysis and the associated distance-
dependent weighting functions is Trapp, J. R. and Doswell III, C. A., 2000:
Radar Data Objective Analysis. J. Atmos. Oceanic Technol., 17, 105--120.

"""

import numpy as np

from scipy.spatial import cKDTree


class Weight(object):
    """
    A class for defining radar data objective analysis parameters and storing
    kd-tree information.

    This class and its kd-tree methods and attributes are particularly useful
    when multiple consecutive radar volumes need to be mapped and each radar
    volume is defined by the same coordinates in space, e.g., similar range
    gates, azimuths, and elevation angles.

    Attributes
    ----------
    func : callable, optional
        Function defining objective analysis weight. The default weighting
        function is an isotropic Barnes distance-dependent weight with constant
        smoothing parameter.
    data_spacing : float, optional
        The data spacing of radar gates in meters used to define the constant
        smoothing parameter of an isotropic Barnes weight. Only applicable if
        the default weighting function is used. For most radar scans this value
        is actually a function of range and elevation, but for most practical
        applications is assumed constant.
    kappa_star : float, optional
        The nondimensional smoothing parameter described in Trapp and Doswell
        (2000). Only applicable for the default weighting function.
    cutoff_radius : float, optional
        The cutoff radius in meters used when querying a kd-tree. A small value
        may increase processing time but produce poor mapping results. See
        SciPy documentation for more information.
    k : int, optional
        The number of nearest neighbours to return when querying a kd-tree. The
        default will return the closest 100 neighbours. Note that for large
        values (e.g., 400+ neighbours), computer memory resources can become
        strained, so caution is advised. However, for typical PPI radar scans,
        this value should, at minimum, be larger than 300.
    leafsize : int
        Leaf size passed to the kd-tree defining the number of points at which
        the algorithm switches over to brute-force. This can affect the
        processing time during the construction and query of the kd-tree, as
        well as the memory required to store the tree. The optimal value
        depends on the nature of the input data. Note that this parameter will
        not affect the results, only the processing time.
    kappa : float, optional
        The constant smoothing parameter described in Trapp and Doswell (2000)
        which defines an isotropic Barnes weight. This parameter is derived
        from the kappa_star and data_spacing parameters. Only applicable for
        the default weighting function.
    distance_weight_vanishes : array_like
        Distance in meters from a radar gate in which the objective analysis
        weighting function effectively vanishes.
    radar_tree : cKDTree
        The cKDTree corresponding to the radar gate locations.
    dists : array_like
        The distances (gate-grid) the k-nearest radar gates are to each grid
        point in the analysis domain computed from querying a kd-tree. These
        distances are not available until the kd-tree is queried.
    inds : array_like
        The indices corresponding to the k-nearest radar gates to each grid
        point in the analysis domain determined from querying a kd-tree. These
        indices are not available until the kd-tree is queried.
    wq : array_like
        The distance-dependent weights corresponding to the k-nearest radar
        gates to each grid point in the analysis domain.

    """

    def __init__(self, func=None, cutoff_radius=np.inf, kappa_star=0.5,
                 data_spacing=1220.0, k=100, leafsize=10):
        """ Initialize. """

        # Default distance-dependent weight parameters
        self.kappa_star = kappa_star
        self.data_spacing = data_spacing
        self.kappa = None

        # Distance-dependent weight function and weights
        self.func = func
        self.wq = None

        # The default weighting function is an isotropic Barnes
        # distance-dependent weight with constant smoothing parameter
        if self.func is None:
            self.kappa = kappa_star * (2.0 * data_spacing)**2
            self.func = lambda r: np.ma.exp(-r**2 / self.kappa)

        # Default kd-tree query parameters
        self.radar_tree = None
        self.k = k
        self.leafsize = leafsize
        self.cutoff_radius = cutoff_radius
        self.dists = None
        self.inds = None

        # Default radar gate locations
        self.z_g = None
        self.y_g = None
        self.x_g = None
        self.z_ref = None
        self.y_ref = None
        self.x_ref = None


    def create_radar_tree(
            self, coords, replace_existing=True, debug=False, verbose=False):
        """
        Create a radar kd-tree instance.

        Parameters
        ----------
        coords : array_like
            The (z, y, x) radar gate locations in meters to be indexed. This
            array is not copied unless this is necessary to produce a
            contiguous array of doubles, so modifying this data will result in
            poor results.
        replace_existing : bool, optional
            True to replace any existing radar kd-tree, False to keep the
            existing kd-tree.
        debug : bool, optional
            True to print debugging information, False to suppress.
        verbose : bool, optional
            True to print progress information, False to suppress.

        """
        if self.radar_tree is None or replace_existing:
            if verbose:
                print 'Creating k-d tree instance for radar gate locations'
            self.radar_tree = cKDTree(coords, leafsize=self.leafsize)

        if debug:
            print 'tree.m = {}'.format(self.radar_tree.m)
            print 'tree.n = {}'.format(self.radar_tree.n)

        return


    def query_tree(self, coords, store=True, debug=False):
        """
        Query a radar gate kd-tree instance to find nearest-neighbour analysis
        grid points.

        Parameters
        ----------
        coords : array_like
            The (z, y, x) analysis grid coordinates in meters to query.
        store : bool, optional
            True to store distance-dependent weights, False otherwise.
        debug : bool, optional
            True to print debugging information, False to suppress.

        """

        # Check for existence of radar kd-tree
        if self.radar_tree is None:
            raise ValueError('No radar kd-tree exists')

        # Query the kd-tree
        dists, inds = self.radar_tree.query(
            coords, k=self.k, p=2.0, eps=0.0,
            distance_upper_bound=self.cutoff_radius)
        if store:
            self.dists = dists
            self.inds = inds

        if debug:
            dist_min, dist_max = dists.min() / 1000.0, dists.max() / 1000.0
            print 'Distance array shape: {}'.format(dists.shape)
            print 'Minimum distance to neighbour: {:.2f} km'.format(dist_min)
            print 'Maximum distance to neighbour: {:.2f} km'.format(dist_max)
            print 'Index array shape: {}'.format(inds.shape)
            print 'Minimum index: {}'.format(inds.min())
            print 'Maximum index: {}'.format(inds.max())

        return dists, inds


    def requery(self, rtol=1.0e-15, atol=20.0, verbose=True):
        """
        Determine whether or not the radar kd-tree needs to be requeried. The
        kd-tree does not need to be requeried if the previous nearest neighbour
        distances and indices were computed from a similar radar volume.

        Parameters
        ----------
        rtol : float, optional
            The relative tolerance used for radar gate location comparisons.
            This value should generally be ignored and can be set very small.
        atol : float, optional
            The absolute tolerance in meters used for radar gate location
            comparisons. Assuming the relative tolerance is very small, then if
            all radar gate comparisons between two independent volumes are
            within this distance apart (e.g., 20 m), then the two volumes are
            classified as being ordered the same way.
        verbose:
            True to print progress information, False to suppress.

        Returns
        -------
        answer : bool
            True if kd-tree needs to be requeried, False otherwise.
        """

        if self.radar_tree is not None:

            # Parse reference radar gate locations and current gate locations
            z_ref, y_ref, x_ref = self.z_ref, self.y_ref, self.x_ref
            z_g, y_g, x_g = self.z_g, self.y_g, self.x_g

            # First check: same number of gates
            if z_ref.size != z_g.size:
                if verbose:
                    print 'Radar volumes have different number of gates'
                return True

            # Second check: similar gate locations and ordering
            x_close = np.allclose(x_ref, x_g, rtol=rtol, atol=atol)
            y_close = np.allclose(y_ref, y_g, rtol=rtol, atol=atol)
            z_close = np.allclose(z_ref, z_g, rtol=rtol, atol=atol)
            if not x_close or not y_close or not z_close :
                if verbose:
                    print 'Radar volumes are ordered differently'
                return True
            else:
                return False

        else:
            if verbose:
                print 'No radar kd-tree exists'
            return True


    def compute_weights(self, dists, store=True):
        """
        Compute distance-dependent weights defined by the specified objective
        analysis weighting function.

        Parameters
        ----------
        dists : array_like
            Gate-grid distances in meters.
        store : bool, optional
            True to store distance-dependent weights, False otherwise. When
            doing any testing of distance-dependent weights, e.g., checking the
            distance the weight vanishes, this parameter should be set to
            False.
        """
        wq = self.func(dists)
        if store:
            self.wq = wq

        return wq

    def compute_distance_weight_vanishes(
            self, rtol=1.0e-15, atol=1.0e-3, verbose=True):
        """
        Determine the distance at which the chosen objective analysis weight
        effectively vanishes. This method is really only meaningful for Barnes
        constant smoothing parameter weighting functions.

        Parameters
        ----------
        rtol : float, optional
            The relative tolerance which defines the weight as vanishing. This
            value can typically be ignored and set to a very small value.
        atol : float, optional
            The absolute tolerance which defines the weight as vanishing.
        verbose : bool, optional
            True to print the vanishing distance, False to suppress.

        """

        # Define test distances and compute distance-dependent weights
        if self.dists is None:
            dists = np.arange(0.0, 15050.0, 50.0)
        else:
            dists = self.dists
        wq = self.compute_weights(dists, store=False)

        # Determine index where weight effectively vanishes
        idx = np.isclose(wq, 0.0, rtol=rtol, atol=atol).argmax()
        self.distance_weight_vanishes = dists[idx]

        if verbose:
            print 'Distance weight vanishes: {:.2f} m'.format(dists[idx])

        return


    def _add_gate_locations(self, coords):
        """
        Add radar gate location coordinates.

        Parameters
        ----------
        coords : list or tuple
            The (z, y, x) radar gate location coordinates.

        """
        self.z_g = coords[0]
        self.y_g = coords[1]
        self.x_g = coords[2]

        return

    def _add_gate_reference(self, coords, replace_existing=False):
        """
        Add reference radar gate location coordinates.

        Parameters
        ----------
        coords : list or tuple
            The (z, y, x) radar gate location coordinates.
        replace_existing : bool, optional
            True to replace existing reference coordinates, False to keep.

        """

        if self.z_ref is None or replace_existing:
            self.z_ref = coords[0]
            self.y_ref = coords[1]
            self.x_ref = coords[2]

        return

    def _reset_gate_reference(self):
        """
        Reset the reference gate coordinates to the current radar gate
        locations.

        """
        self.z_ref = self.z_g.copy()
        self.y_ref = self.y_g.copy()
        self.x_ref = self.x_g.copy()

        return
