"""
grid.interp.mapper
==================

"""

import numpy as np

from scipy import spatial
from mpl_toolkits.basemap import pyproj

from pyart.config import get_fillvalue, get_field_name
from pyart.core import Grid


def grid_radar(
        radar, grid_coords, grid_origin=None, fields=None,
        weighting_function='Cressman', toa=17000.0,
                      leafsize=10.0, k=100, eps=0.0, roi_func='constant',
                      constant_roi=1000.0, cutoff_radius=5000.0,
                      min_radius=250.0, x_factor=0.01, y_factor=0.01,
                      z_factor=0.01, nb=1.0, bsp=1.0, h_factor=0.1,
                      smooth_func='constant', kappa_star=0.5,
                      data_space=1220.0, map_roi=False, map_dist=True,
                      proj='lcc', datum='NAD83', ellps='GRS80',
                      fill_value=None, debug=False):
    """
    Map scanning radar to a Cartesian analysis domain.

    Parameters
    ----------
    radar : Radar
        A radar object to be mapped to the Cartesian analysis grid.
    grid_coords : tuple or list
        The (z, y, x) coordinates of the grid in meters. These can describe
        either a uniform or non-uniform grid.
    grid_origin : tuple or list
        The (latitude, longitude) location of the analysis domain origin. The
        default (None) will use the radar latitude and longitude as the center
        of the analysis domain.
    fields : list
        List of radar fields which will be mapped to the Cartesian analysis
        domain. The default (None) will map all the radar fields.
    weighting_function : 'nearest' or 'Barnes' or 'Cressman'
        Type of weighting function used to weight collected neighbors. Note
        that if 'nearest', then only the closest neighbor is used, and no
        weighting is performed.
    toa : float
        The "top of the atmosphere" in meters. Collected neighbors above this
        height are excluded.
    leafsize : int
        Leaf size passed to the cKDTree object. This can affect the processing
        time during the construction and query of the cKDTree, as well as the
        memory required to store the tree. The optimal value depends on the
        nature of the input data. Note that this parameter will not affect
        the results, only the processing time.
    k : int
        The number of nearest neighbors to return. When weighting_function
        is 'Barnes' or 'Cressman', k >> 1. If weighting_function is 'nearest',
        then this parameter can be ignored since k = 1 automatically. Similar
        to the leafsize parameter, k will affect the processing time during
        the cKDTree query (but not its construction), as well as memory usage.
        When specifiying k, the following two properties should be satisfied
        (1) all relevent neighborhood points are collected (depends on cutoff
        radius or radius of influence) and (2) memory is managed
        appropriately. Through sensitivity experiments, it was determined that
        for volumetric radar data, 50 < k < 600.
    eps : float
        Return approximate nearest neighbors. The k-th returned value is
        guaranteed to be no further than (1 + eps) times the distance to
        the real k-th nearest neighbor.
    cutoff_radius : float
        The largest radius in meters to search for points. This is only valid
        when 'weighting_function' is 'Barnes', and should be large enough to
        capture all points where the Barnes weight is nonzero.
    proj : str
        See pyproj documentation for more information.
    datum : str
        See pyproj documentation for more information.
    ellps : str
        See pyproj documentation for more information.
    debug : bool
        True to print debugging information, False to suppress.

    Return
    ------
    grid : Grid
        Grid object with mapped radar fields, axes, and metadata.
    """

    # Parse missing value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Check weight function parameters
    if weighting_function not in ['Cressman', 'Barnes', 'nearest']:
        raise ValueError('Unsupported weighting_function')
    if weighting_function in ['Cressman', 'Barnes'] and k == 1:
        raise ValueError('For Cressman or Barnes weights, k > 1')

    # Get grid origin if not given
    if grid_origin is None:
        lat0 = radar.latitude['data'][0]
        lon0 = radar.longitude['data'][0]
        grid_origin = (lat0, lon0)

    # Get fields which should be mapped
    # If no fields are given, then all fields will be mapped
    if fields is None:
        fields = radar.fields.keys()
    else:
        if not set(fields).issubset(set(radar.fields.keys())):
            raise ValueError('One or more specified fields do not exist')

    # Calculate radar offset from the origin
    offset = _calculate_radar_offset(radar, grid_origin, proj=proj,
                        datum=datum, ellps=ellps, debug=debug)

    # Calculate Cartesian locations of radar gates relative to grid origin
    z_g, y_g, x_g = _radar_coords_to_cartesian(radar, debug=debug)
    z_g = z_g + offset[0]
    y_g = y_g + offset[1]
    x_g = x_g + offset[2]

    # Parse Cartesian coordinates of analysis grid
    z_a, y_a, x_a = grid_coords
    nz = len(z_a)
    ny = len(y_a)
    nx = len(x_a)
    if debug:
        print 'Grid shape is nz = %i, ny = %i, nx = %i' % (nz, ny, nx)

    # Create analysis grid mesh
    z_a, y_a, x_a = np.meshgrid(z_a, y_a, x_a, indexing='ij')
    z_a = z_a.flatten()
    y_a = y_a.flatten()
    x_a = x_a.flatten()

    if debug:
        print 'Grid array has shape %s' % (z_a.shape,)

    # Compute the radius of influence for each analysis point, if necessary
    if weighting_function == 'Cressman':
        if roi_func == 'constant':
            roi = constant_roi

        elif roi_func == 'dist':
            roi = default_roi_func_dist(
                    x_a, y_a, z_a, offset, x_factor=x_factor,
                    y_factor=y_factor, z_factor=z_factor,
                    min_radius=min_radius)

        elif roi_func == 'beam':
            roi = default_roi_func_beam(
                    x_a, y_a, z_a, offset, nb=nb, bsp=bsp,
                    h_factor=h_factor, min_radius=min_radius)

        elif hasattr(roi_func, '__call__'):
            roi = roi_func(x_a, y_a, z_a, offset)

        else:
            raise ValueError('Unsupported roi_func')

        # Compute the maximum radius of influece within the analysis domain
        # This will serve to "prune" the k-d tree search when a Cressman
        # scheme is desired
        max_roi = np.max(roi)

        if debug:
            print 'Minimum ROI is %.2f m' % np.min(roi)
            print 'Maximum ROI is %.2f m' % np.max(roi)
            print 'ROI array has shape %s' % (np.shape(roi),)

    # Compute the smoothing parameter for each analysis point, if necessary
    if weighting_function == 'Barnes':
        if smooth_func == 'constant':
            kappa = kappa_star * (2.0 * data_space)**2

        elif hasattr(smooth_func, '__call__'):
            kappa = smooth_func(x_a, y_a, z_a, offset)

        else:
            raise ValueError('Unsupported smooth_func')

    # Remove radar gates that are past the "top of the atmosphere"
    # This will speed up processing time during the creation of the k-d tree
    # since it removes unneccessary gates
    is_below_toa = z_g <= toa
    if debug:
        print 'Number of radar gates below TOA is %i' % is_below_toa.sum()

    # Remove radar gates that are too far from the analysis grid to be
    # captured by any analysis grid point radius of influence (Cressman)
    # or the cutoff radius (Barnes)
    if weighting_function in ['Cressman', 'Barnes']:
        # Compute the distance each radar gate is from the analysis domain
        # origin and the maximum distance any analysis point is from the
        # analysis domain origin
        dist_g = np.sqrt(x_g**2 + y_g**2 + z_g**2)
        max_dist_a = np.sqrt(x_a**2 + y_a**2 + z_a**2).max()

        if weighting_function == 'Barnes':
            is_captured = dist_g <= max_dist_a + cutoff_radius
        else:
            is_captured = dist_g <= max_dist_a + max_roi

        is_valid_gate = np.logical_and(is_below_toa, is_captured)

    else:
        is_valid_gate = is_below_toa

    if debug:
        sum_pruned = is_valid_gate.sum()
        print 'Total number of radar gates before pruning: %i' % z_g.size
        print 'Total number of radar gates after pruning: %i' % sum_pruned

    # Update radar gates and radar data
    z_g = z_g[is_valid_gate]
    y_g = y_g[is_valid_gate]
    x_g = x_g[is_valid_gate]
    for field in fields:
        radar.fields[field].update({'data':
                    radar.fields[field]['data'].flatten()[is_valid_gate]})

    # Create k-d tree object for radar gate locations
    # Depending on the number of radar gates this can be resource intensive
    # but should nonetheless take on the order of seconds to create
    tree_g = spatial.cKDTree(zip(z_g, y_g, x_g), leafsize=leafsize)

    if debug:
        print 'tree.m = %i, tree.n = %i' % (tree_g.m, tree_g.n)

    # Query k-d tree
    if weighting_function == 'nearest':
        dist, ind = tree_g.query(zip(z_a, y_a, x_a), k=1, p=2.0, eps=eps,
                                 distance_upper_bound=np.inf)
    elif weighting_function == 'Barnes':
        dist, ind = tree_g.query(zip(z_a, y_a, x_a), k=k, p=2.0, eps=eps,
                                 distance_upper_bound=cutoff_radius)

        # Compute the Barnes distance-dependent weights
        dist = np.ma.masked_invalid(dist)
        wq = np.ma.exp(-dist**2 / kappa)

    else:
        dist, ind = tree_g.query(zip(z_a, y_a, x_a), k=k, p=2.0, eps=eps,
                                 distance_upper_bound=max_roi)

        # Compute the Cressman distance-dependent weights
        # Where the neighbor is further than the Cressman radius of
        # influence, set its weight to zero
        roi_stack = np.repeat(roi, k).reshape(roi.size, k)
        wq = (roi_stack**2 - dist**2) / (roi_stack**2 + dist**2)
        is_past_roi = dist > roi_stack
        wq[is_past_roi] = 0.0

    if debug:
        print 'Distance array has shape %s' % (dist.shape,)
        print 'Minimum distance is %.2f m' % dist.min()
        print 'Maximum distance is %.2f m' % dist.max()
        print 'Index array has shape %s' % (ind.shape,)
        print 'Minimum index is %i' % ind.min()
        print 'Maximum index is %i' % ind.max()

    # Missing neighbors are indicated with an index set to tree.n
    # This condition will not be met for the nearest neighbor scheme, but
    # it can be met for the Cressman and Barnes schemes
    # We can safely set the index of missing neighbors to 0 since
    # its weight has already been set to 0 later and thus it will not affect
    # the interpolation (weighted averaging)
    bad_index = ind == tree_g.n
    ind[bad_index] = 0

    # Interpolate the radar data onto the analysis grid and populate the
    # mapped fields dictionary
    map_fields = {}
    for field in fields:
        if debug:
            print 'Mapping field: %s' % field

        # Get radar data
        radar_data = radar.fields[field]['data']

        if weighting_function == 'nearest':
            fq = radar_data[ind]

        else:
            # Compute the distance-weighted average
            # This is applicable for both Cressman and Barnes schemes
            fq = np.ma.average(radar_data[ind], weights=wq, axis=1)

        map_fields[field] = {'data': fq.reshape(nz, ny, nx)}

        # Populate mapped field metadata
        [map_fields[field].update({meta: value}) for meta, value in
         radar.fields[field].iteritems() if meta != 'data']

    # Map the nearest neighbor distances, if necessary
    if map_dist:
        field = 'nearest_neighbor_distance'
        map_fields[field] = {
            'units': 'meters',
            'long_name': 'Distance to closest radar gate'}
        if weighting_function == 'nearest':
            map_fields[field]['data'] = dist.reshape(nz, ny, nx)
        else:
            map_fields[field]['data'] = dist.min(axis=1).reshape(nz, ny, nx)

    # Map the radius of influence, if necessary
    if map_roi and 'roi' in locals():
        field = 'radius_of_influence'
        map_fields[field] = {
            'data': roi.reshape(nz, ny, nx),
            'units': 'meters',
            'long_name': 'Radius of influence used for mapping'}

    # Populate Cartesian axes dictionaries
    x_disp = {
        'data': grid_coords[2],
        'units': 'meters',
        'axis': 'X',
        'long_name': 'x-coordinate in Cartesian system'}

    y_disp = {
        'data': grid_coords[1],
        'units': 'meters',
        'axis': 'Y',
        'long_name': 'y-coordinate in Cartesian system'}

    z_disp = {
        'data': grid_coords[0],
        'units': 'meters',
        'axis': 'Z',
        'positive': 'up',
        'long_name': 'z-coordinate in Cartesian system'}

    # Populate grid origin dictionaries
    altitude = {
        'data': np.array([0.0]),
        'units': 'meters',
        'standard_name': 'altitude',
        'long_name': 'Altitude at grid origin above mean sea level'}

    latitude = {
        'data': np.array([grid_origin[0]]),
        'units': 'degrees_N',
        'standard_name': 'latitude',
        'valid_min': -90.0,
        'valid_max': 90.0,
        'long_name': 'Latitude at grid origin'}

    longitude = {
        'data': np.array([grid_origin[1]]),
        'units': 'degrees_E',
        'standard_name': 'longitude',
        'valid_min': -180.0,
        'valid_max': 180.0,
        'long_name': 'Longitude at grid origin'}

    # Populate time dictionaries
    time = {
        'data': np.array([radar.time['data'].min()]),
        'units': radar.time['units'],
        'calendar': radar.time['calendar'],
        'standard_name': radar.time['standard_name'],
        'long_name': 'Time in seconds since volume start'}

    time_start = {
        'data': np.array([radar.time['data'].min()]),
        'units': radar.time['units'],
        'calendar': radar.time['calendar'],
        'standard_name': radar.time['standard_name'],
        'long_name': 'Time in seconds since volume start'}

    time_end = {
        'data': np.array([radar.time['data'].max()]),
        'units': radar.time['units'],
        'calendar': radar.time['calendar'],
        'standard_name': radar.time['standard_name'],
        'long_name': 'Time in seconds since volume end'}

    # Create axes
    axes = {
        'time': time,
        'time_start': time_start,
        'time_end': time_end,
        'x_disp': x_disp,
        'y_disp': y_disp,
        'z_disp': z_disp,
        'alt': altitude,
        'lat': latitude,
        'lon': longitude}

    # Create metadata
    metadata = {
        'process_version': '',
        'references': '',
        'Conventions': '',
        'site': '',
        'facility_id': '',
        'project': '',
        'state': '',
        'comment': '',
        'institution': '',
        'country': '',
        'title': 'Mapped Moments to Cartesian Grid'}

    # Add radar-specific metadata to grid metadata
    metadata['radar_0_altitude'] = radar.altitude['data'][0]
    metadata['radar_0_latitude'] = radar.latitude['data'][0]
    metadata['radar_0_longitude'] = radar.longitude['data'][0]
    metadata['radar_0_instrument_name'] = radar.metadata['instrument_name']

    return Grid(map_fields, axes, metadata)


def default_roi_func_dist(x_a, y_a, z_a, offset, x_factor=1.0, y_factor=1.0,
                          z_factor=1.0, min_radius=250.0):
    """
    """

    # Apply the offset to the analysis grid such that it is now
    # radar-centric
    z_a = z_a - offset[0]
    y_a = y_a - offset[1]
    x_a = x_a - offset[2]

    roi = np.sqrt(x_factor * x_a**2 + y_factor * y_a**2 +
                  z_factor * z_a**2) + min_radius

    return roi


def default_roi_func_beam(x_a, y_a, z_a, offset, nb=1.0, bsp=1.0,
                          h_factor=1.0, min_radius=250.0):
    """
    """

    # Apply the offset to the analysis grid such that it is now
    # radar-centric
    z_a = z_a - offset[0]
    y_a = y_a - offset[1]
    x_a = x_a - offset[2]

    roi = h_factor * z_a + np.sqrt(x_a**2 + y_a**2) * \
          np.tan(nb * bsp * np.pi / 180.0) + min_radius

    return roi
