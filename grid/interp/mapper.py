"""
grid.interp.mapper
==================

"""

import numpy as np

from scipy.spatial import cKDTree

from pyart.config import get_fillvalue, get_field_name, get_metadata
from pyart.core import Grid

from ..util import transform
from ..core import Weight

# TODO: smartly interpolate the time information of the radar to analysis grid
# points. Note that the time information for a typical radar is the average or
# median time for each ray, i.e., since the sampling of the radar is very quick
# it is overkill to record the time information at each range gate


def grid_radar(
        radar, grid_coords, weight=None, lat_0=None, lon_0=None, alt_0=None,
        fields=None, toa=17000.0, max_range=None, proj='lcc', datum='NAD83',
        ellps='GRS80', fill_value=None, sqi_field=None, dist_field=None,
        time_field=None, gqi_field=None, debug=False, verbose=False):
    """
    Grid (map) a scanning radar volume to a specified 3-D Cartesian domain.
    This routine uses a k-d tree space-partitioning data structure for the
    efficient searching of the k-nearest neighbours. Uniform and non-uniform
    grids are both accepted.

    Parameters
    ----------
    radar : Radar
        Py-ART radar object containing the fields to be mapped.
    grid_coords : list or tuple
        The (z, y, x) Cartesian coordinates of the analysis domain the radar
        data will be mapped onto. See the lat_0, lon_0, and alt_0 parameters
        in order to specify the origin of the analysis domain.
    weight : Weight, optional
        Weight object defining the radar data objective analysis parameters
        and storing any available kd-tree information. Default uses an
        isotropic distance-dependent Barnes weight with a constant smoothing
        paramter.
    lat_0 : float, optional
        The latitude of the grid origin. The default uses the radar's latitude
        as the grid origin.
    lon_0 : float, optional
        The longitude of the grid origin. The default uses the radar's
        longitude as the grid origin.
    alt_0 : float, optional
        The altitude above mean sea level of the grid origin. The default uses
        the radar's altitude as the grid origin.
    fields : str or list or tuple, optional
        The list of radar fields to be gridded. The default grids all available
        radar fields.
    toa : float, optional
        Top of the atmosphere in meters. Radar gates above this altitude are
        ignored. Lower heights may increase processing time substantially but
        may also produce poor results if this value is similar or lower than
        the top of the grid domain.
    max_range : float, optional

    proj : str, optional

    datum : str, optional

    ellps : str, optional


    Return
    ------
    grid : Grid
        Py-ART Grid containing the mapped scanning radar data, axes
        information, and metadata.

    """

    # Parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse field names
    if sqi_field is None:
        sqi_field = get_field_name('normalized_coherent_power')
    if dist_field is None:
        dist_field = get_field_name('nearest_neighbor_distance')
    if gqi_field is None:
        gqi_field = get_field_name('grid_quality_index')
    if time_field is None:
        time_field = 'radar_sampling_time'

    # Parse analysis domain origin
    if lat_0 is None:
        lat_0 = radar.latitude['data'][0]
    if lon_0 is None:
        lon_0 = radar.longitude['data'][0]
    if alt_0 is None:
        alt_0 = radar.altitude['data'][0]

    # Parse fields to map
    if fields is None:
        fields = radar.fields.keys()
    if isinstance(fields, str):
        fields = [fields]

    # Parse radar data objective analysis weight
    if weight is None:
        weight = Weight(radar)

    # Parse maximum range
    if max_range is None:
        max_range = radar.range['data'].max()

    # Calculate radar offset relative to the analysis domain origin
    offset = transform._calculate_radar_offset(
        radar, lat_0=lat_0, lon_0=lon_0, alt_0=alt_0, proj=proj, datum=datum,
        ellps=ellps, debug=debug, verbose=verbose)

    # Compute Cartesian coordinates of radar gates relative to specified origin
    # Add reference gate locations and current gate locations to weight object
    # which will help determine if the kd-tree needs to be requeried or not
    z_g, y_g, x_g = transform.standard_refraction(
        radar, offset=offset, debug=debug, verbose=verbose)
    weight._add_gate_reference([z_g, y_g, x_g], replace_existing=False)
    weight._add_gate_locations([z_g, y_g, x_g])

    if debug:
        N = z_g.size
        print 'Number of radar gates before pruning: {}'.format(N)

    # Do not consider radar gates that are past the "top of the atmosphere"
    # This will speed up processing time during the creation and querying of
    # the k-d tree since it removes unneccessary gates
    is_below_toa = z_g <= toa

    if debug:
        N = np.count_nonzero(is_below_toa)
        print 'Number of radar gates below TOA: {}'.format(N)

    # Slice radar coordinates below the TOA
    z_g = z_g[is_below_toa]
    y_g = y_g[is_below_toa]
    x_g = x_g[is_below_toa]

    # Slice radar data fields below the TOA but preserve original radar data
    radar_data = {}
    for field in fields:
        data = radar.fields[field]['data'].copy().flatten()
        radar_data[field] = data[is_below_toa]

    # Parse Cartesian coordinates of analysis domain
    z_a, y_a, x_a = grid_coords
    nz, ny, nx = len(z_a), len(y_a), len(x_a)

    if debug:
        print 'Grid z-size: {}'.format(nz)
        print 'Grid y-size: {}'.format(ny)
        print 'Grid x-size: {}'.format(nx)

    # Create analysis domain coordinates mesh
    z_a, y_a, x_a = np.meshgrid(z_a, y_a, x_a, indexing='ij')
    z_a, y_a, x_a = z_a.flatten(), y_a.flatten(), x_a.flatten()

    if debug:
        print 'Grid 1-D array shape: {}'.format(z_a.shape)

    # Query the radar gate k-d tree for the k-nearest analysis grid points.
    # Also compute the distance-dependent weights
    # This is the step that consumes the most processing time, but it can be
    # skipped if results from a similar radar volume have already computed and
    # stored in the weight object
    if weight.requery(verbose=verbose):

        # Create k-d tree object from radar gate locations
        # Depending on the number of radar gates this can be resource intensive
        # but nonetheless should take on the order of 1 second to create
        weight.create_radar_tree(
            zip(z_g, y_g, x_g), replace_existing=True, debug=debug,
            verbose=verbose)

        if verbose:
            print 'Querying k-d tree for the k-nearest analysis grid points'

        dists, inds = weight.query_tree(
            zip(z_a, y_a, x_a), store=True, debug=debug)

        # Compute distance-dependent weights
        if verbose:
            print 'Computing distance-dependent weights'
        wq = weight.compute_weights(dists)

        # Reset reference radar gate coordinates
        weight._reset_gate_reference()

    else:
        dists, inds, wq = weight.dists, weight.inds, weight.wq

    # Missing neighbors are indicated with an index set to tree.n
    # This condition will not be met for the nearest neighbor scheme, but
    # it can be met for the Cressman and Barnes schemes if the cutoff radius
    # is not large enough
    is_bad_index = inds == weight.radar_tree.n

    if debug:
        N = np.count_nonzero(is_bad_index)
        print 'Number of invalid indices: {}'.format(N)

    # Analysis grid points which are further than the specified maximum
    # range away from the radar should not be interpolated
    # This is to account for the unambiguous range of the radar, i.e., not all
    # analysis grid points should be considered for interpolation because the
    # radar does not have observations out to all distances
    z_r, y_r, x_r = offset
    _range = np.sqrt((z_a - z_r)**2 + (y_a - y_r)**2 + (x_a - x_r)**2)
    is_far = _range > max_range

    if debug:
        N = np.count_nonzero(is_far)
        print 'Number of analysis points too far from radar: {}'.format(N)

    # Interpolate the radar data onto the analysis domain grid
    # Populate the mapped fields dictionary
    map_fields = {}
    for field in fields:

        if verbose:
            print 'Mapping radar field: {}'.format(field)

        # Populate field metadata
        map_fields[field] = get_metadata(field)

        # Compute distance-dependent weighted average of radar field
        # Mask analysis grid points further than the maximum (unambiguous)
        # range
        fq = np.ma.average(radar_data[field][inds], weights=wq, axis=1)
        fq = np.ma.masked_where(is_far, fq, copy=False)
        fq.set_fill_value(fill_value)

        # Save interpolated radar field
        map_fields[field]['data'] = fq.reshape(nz, ny, nx).astype(np.float32)

    # Create grid quality index (GQI) from radar signal quality index SQI),
    # which should have a value between [0, 1] at every radar gate
    if sqi_field in radar.fields:

        if verbose:
            print 'Mapping GQI field: {}'.format(sqi_field)

        # Parse SQI data
        sqi = radar.fields[sqi_field]['data'].copy().flatten()
        sqi = sqi[is_below_toa][inds]

        # Compute distance-dependent weighted average of SQI field and save
        # results
        fq = np.average(np.ma.filled(sqi, 0.0), weights=wq, axis=1)
        fq = np.where(is_far, 0.0, fq)
        map_fields[gqi_field] = {
            'data': fq.reshape(nz, ny, nx).astype(np.float32),
            'standard_name': gqi_field,
            'long_name': 'Grid quality index',
            '_FillValue': None,
            'units': 'unitless',
            'valid_min': 0.0,
            'valid_max': 1.0,
            'comment': '0 = minimum grid quality, 1 = maximum grid quality',
            }

    # Save the nearest-neighbour (gate-grid) distances
    map_fields[dist_field] = {
        'data': dists.min(axis=1).reshape(nz, ny, nx).astype(np.float32),
        'standard_name': dist_field,
        'long_name': 'Nearest neighbor distance',
        'valid_min': 0.0,
        '_FillValue': None,
        'units': 'meters',
        'comment': 'Distance to closest radar gate',
        }

    # Interpolate radar time data
    # Analysis grid points are assigned the median time of the k-nearest radar
    # gates
    time = np.repeat(radar.time['data'], radar.ngates).reshape(
        radar.nrays, radar.ngates).flatten()
    fq = np.median(time[is_below_toa][inds], axis=1)
    map_fields[time_field] = {
        'data': fq.reshape(nz, ny, nx).astype(np.float32),
        'standard_name': 'radar_sampling_time',
        'long_name': 'Radar sampling time',
        '_FillValue': None,
        'units': radar.time['units'],
        'calendar': radar.time['calendar'],
        }

    # Create grid axes
    axes = _populate_axes(
        radar, grid_coords, lat_0=lat_0, lon_0=lon_0, alt_0=alt_0)

    # Create grid metadata
    metadata = _populate_metadata(radar)

    return Grid(map_fields, axes, metadata)


def _grid_radar_nearest(
        radar, grid_coords, lat_0=None, lon_0=None, alt_0=None, fields=None,
        toa=17000.0, max_range=None, leafsize=10, eps=0.0, proj='lcc',
        datum='NAD83', ellps='GRS80', fill_value=None, sqi_field=None,
        gqi_field=None, dist_field=None, time_field=None, debug=False,
        verbose=False):
    """
    Map scanning radar data to a Cartesian analysis domain using nearest
    neighbour scheme.

    Parameters
    ----------
    radar : Radar
        Py-ART radar object containing the fields to be mapped.
    grid_coords : list or tuple
        The (z, y, x) Cartesian coordinates of the analysis domain the radar
        data will be mapped onto. See the lat_0, lon_0, and alt_0 parameters
        in order to specify the origin of the analysis domain.
    lat_0 : float, optional
        The latitude of the grid origin. The default uses the radar's latitude
        as the grid origin.
    lon_0 : float, optional
        The longitude of the grid origin. The default uses the radar's
        longitude as the grid origin.
    alt_0 : float, optional
        The altitude above mean sea level of the grid origin. The default uses
        the radar's altitude as the grid origin.
    fields : str or list or tuple, optional
        The list of radar fields to be gridded. The default grids all available
        radar fields.
    toa : float, optional
        Top of the atmosphere in meters. Radar gates above this altitude are
        ignored. Lower heights may increase processing time substantially but
        may also produce poor results if this value is similar or lower than
        the top of the grid domain.
    max_range : float
        The "unambiguous range" of the radar. Analysis grid points which are
        not within this range from the radar are excluded.
    leafsize : int
        Leaf size passed to the cKDTree object. This can affect the processing
        time during the construction and query of the cKDTree, as well as the
        memory required to store the tree. The optimal value depends on the
        nature of the input data. Note that this parameter will not affect
        the results, only the processing time.
    eps : float
        Return approximate nearest neighbors. The k-th returned value is
        guaranteed to be no further than (1 + eps) times the distance to
        the real k-th nearest neighbor.
    proj : str, optional
        See pyproj documentation for more information.
    datum : str
        See pyproj documentation for more information.
    ellps : str
        See pyproj documentation for more information.
    debug : bool
        True to print debugging information, False to suppress.
    verbose : bool
        True to print progress information, False to suppress.

    Return
    ------
    grid : Grid
        Py-ART grid object containing the mapped scanning radar data, axes
        information, and metadata.

    """

    # Parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse field names
    if dist_field is None:
        dist_field = 'nearest_neighbor_distance'
    if time_field is None:
        time_field = 'radar_sampling_time'

    # Parse analysis domain origin
    if lat_0 is None:
        lat_0 = radar.latitude['data'][0]
    if lon_0 is None:
        lon_0 = radar.longitude['data'][0]
    if alt_0 is None:
        alt_0 = radar.altitude['data'][0]

    # Parse maximum range
    if max_range is None:
        max_range = radar.range['data'].max()

    # Parse fields to map
    if fields is None:
        fields = radar.fields.keys()

    # Calculate radar offset relative to the analysis domain origin
    offset = transform._calculate_radar_offset(
        radar, lat_0=lat_0, lon_0=lon_0, alt_0=alt_0, proj=proj, datum=datum,
        ellps=ellps, debug=debug, verbose=verbose)

    # Compute Cartesian coordinates of radar gates and apply origin offset
    z_g, y_g, x_g = transform.standard_refraction(
        radar, offset=offset, debug=debug, verbose=verbose)

    if debug:
        N = z_g.size
        print 'Number of radar gates before pruning: {}'.format(N)

    # Do not consider radar gates that are past the "top of the atmosphere"
    # This will speed up processing time during the creation of the k-d tree
    # since it removes unneccessary gates
    is_below_toa = z_g <= toa

    if debug:
        n = is_below_toa.sum()
        print 'Number of radar gates below TOA: {}'.format(n)

    # Slice radar coordinates below the TOA
    z_g = z_g[is_below_toa]
    y_g = y_g[is_below_toa]
    x_g = x_g[is_below_toa]

    # Slice radar data fields below the TOA but preserve original radar data
    radar_data = {}
    for field in fields:
        data = radar.fields[field]['data'].copy().flatten()
        radar_data[field] = data[is_below_toa]

    # Create k-d tree object for radar gate locations
    # Depending on the number of radar gates this can be resource intensive
    # but nonetheless should take on the order of 1 second to create
    if debug:
        print 'Creating k-d tree instance for radar gate locations'
    tree_g = cKDTree(zip(z_g, y_g, x_g), leafsize=leafsize)

    if debug:
        print 'tree.m = {}'.format(tree_g.m)
        print 'tree.n = {}'.format(tree_g.n)

    # Parse Cartesian coordinates of analysis domain grid
    z_a, y_a, x_a = grid_coords
    nz, ny, nx = z_a.size, y_a.size, x_a.size

    if debug:
        print 'Grid z-size: {}'.format(nz)
        print 'Grid y-size: {}'.format(ny)
        print 'Grid x-size: {}'.format(nx)

    # Create analysis domain coordinates mesh
    z_a, y_a, x_a = np.meshgrid(z_a, y_a, x_a, indexing='ij')
    z_a, y_a, x_a = z_a.flatten(), y_a.flatten(), x_a.flatten()

    if debug:
        print 'Grid 1-D array shape: {}'.format(z_a.shape)

    # Query the radar gate k-d tree for the nearest analysis grid points
    # This is the step that consumes the most processing time
    if debug:
        print 'Querying k-d tree for the k-nearest analysis grid points'
    dist, ind = tree_g.query(
        zip(z_a, y_a, x_a), k=1, p=2.0, eps=eps, distance_upper_bound=np.inf)

    if debug:
        dist_min, dist_max = dist.min() / 1000.0, dist.max() / 1000.0
        print 'Distance array shape: {}'.format(dist.shape)
        print 'Minimum gate-grid distance: {:.2f} km'.format(dist_min)
        print 'Maximum gate-grid distance: {:.2f} km'.format(dist_max)
        print 'Index array shape: {}'.format(ind.shape)
        print 'Minimum index: {}'.format(ind.min())
        print 'Maximum index: {}'.format(ind.max())

    # Analysis grid points which are further than the specified maximum
    # range away from the radar should not be interpolated
    # This is to account for the unambiguous range of the radar, i.e., not all
    # analysis grid points should be considered for interpolation because the
    # radar does not have observations at all distances
    z_r, y_r, x_r = offset
    _range = np.sqrt((z_a - z_r)**2 + (y_a - y_r)**2 + (x_a - x_r)**2)
    is_far = _range > max_range

    if debug:
        n = is_far.sum()
        print 'Number of analysis points too far from radar: {}'.format(n)

    # Interpolate the radar field data onto the analysis domain grid
    # Populate the mapped fields dictionary
    map_fields = {}
    for field in fields:
        if debug:
            print 'Mapping radar field: {}'.format(field)

        # Populate field metadata
        map_fields[field] = get_metadata(field)

        # Nearest neighbours
        # Mask analysis grid points further than the maximum (unambiguous)
        # range
        fq = radar_data[field][ind]
        fq = np.ma.masked_where(is_far, fq, copy=False)

        # Save interpolated data
        map_fields[field]['data'] = fq.reshape(nz, ny, nx).astype(np.float32)

    # Save the nearest neighbor distances
    map_fields[dist_field] = {
        'data': dist.reshape(nz, ny, nx).astype(np.float32),
        'standard_name': 'nearest_neighbor_distance',
        'long_name': 'Nearest neighbor distance',
        'valid_min': 0.0,
        'valid_max': np.inf,
        'units': 'meters',
        '_FillValue': None,
        'comment': '',
        }

    # Save radar time data
    # Analysis grid points are assigned the time of the closest radar gate
    time = np.repeat(radar.time['data'], radar.ngates).reshape(
        radar.nrays, radar.ngates).flatten()
    fq = time[is_below_toa][ind]
    map_fields[time_field] = {
        'data': fq.reshape(nz, ny, nx).astype(np.float32),
        'standard_name': 'radar_sampling_time',
        'long_name': 'Radar sampling time',
        '_FillValue': None,
        'units': radar.time['units'],
        'calendar': radar.time['calendar'],
        }

    # Create grid axes
    axes = _populate_axes(
        radar, grid_coords, lat_0=lat_0, lon_0=lon_0, alt_0=alt_0)

    # Create grid metadata
    metadata = _populate_metadata(radar)

    return Grid(map_fields, axes, metadata)


def _populate_axes(radar, grid_coords, lat_0=None, lon_0=None, alt_0=0.0):
    """
    Populate grid axes including metadata.

    Parameters
    ----------
    radar : Radar

    Returns
    -------
    axes : dict
        Dictionary containing axes information including metadata.

     """

    # Populate Cartesian axes
    x_disp = {
        'data': grid_coords[2].astype(np.float32),
        'standard_name': 'x_coordinate',
        'long_name': 'x-coordinate in Cartesian system',
        'units': 'meters',
        'axis': 'X',
        'positive': 'east',
        }
    y_disp = {
        'data': grid_coords[1].astype(np.float32),
        'standard_name': 'y_coordinate',
        'long_name': 'y-coordinate in Cartesian system',
        'units': 'meters',
        'axis': 'Y',
        'positive': 'north',
        }
    z_disp = {
        'data': grid_coords[0].astype(np.float32),
        'standard_name': 'z_coordinate',
        'long_name': 'z-coordinate in Cartesian system',
        'units': 'meters',
        'axis': 'Z',
        'positive': 'up',
        }

    # Populate grid origin locations
    altitude = {
        'data': np.array([alt_0], dtype=np.float32),
        'standard_name': 'altitude',
        'long_name': 'Altitude at grid origin above mean sea level',
        'units': 'meters',
        }
    latitude = {
        'data': np.array([lat_0], dtype=np.float32),
        'standard_name': 'latitude',
        'long_name': 'Latitude at grid origin',
        'units': 'degrees_N',
        'valid_min': -90.0,
        'valid_max': 90.0,
        }
    longitude = {
        'data': np.array([lon_0], dtype=np.float32),
        'standard_name': 'longitude',
        'long_name': 'Longitude at grid origin',
        'units': 'degrees_E',
        'valid_min': -180.0,
        'valid_max': 180.0,
        }

    # Populate time information
    time = {
        'data': np.array([radar.time['data'].min()], dtype=np.float64),
        'standard_name': 'time',
        'long_name': 'Time in seconds since radar volume start',
        'units': radar.time['units'],
        'calendar': 'gregorian',
        }
    time_start = {
        'data': np.array([radar.time['data'].min()], dtype=np.float64),
        'standard_name': 'time_start',
        'long_name': 'Time in seconds since radar volume start',
        'units': radar.time['units'],
        'calendar': 'gregorian',
        }
    time_end = {
        'data': np.array([radar.time['data'].max()], dtype=np.float64),
        'standard_name': 'time_end',
        'long_name': 'Time in seconds since radar volume end',
        'units': radar.time['units'],
        'calendar': 'gregorian',
        }

    return {
        'time': time,
        'time_start': time_start,
        'time_end': time_end,
        'x_disp': x_disp,
        'y_disp': y_disp,
        'z_disp': z_disp,
        'alt': altitude,
        'lat': latitude,
        'lon': longitude,
        }


def _populate_metadata(radar):
    """
    """
    # Datastreams attribute
    datastream_description = (
        'A string consisting of the datastream(s), datastream version(s), '
        'and datastream date (range).')

    return {
        'process_version': '',
        'references': '',
        'Conventions': '',
        'site_id': '',
        'site': '',
        'facility_id': '',
        'project': '',
        'state': '',
        'comment': '',
        'institution': '',
        'country': '',
        'description': '',
        'title': 'Mapped Moments to Cartesian Grid',
        'project': '',
        'input_datastreams_description': datastream_description,
        'input_datastreams_num': 1,
        'input_datastreams': '',
        'radar_0_altitude': radar.altitude['data'][0],
        'radar_0_latitude': radar.latitude['data'][0],
        'radar_0_longitude': radar.longitude['data'][0],
        'radar_0_instrument_name': radar.metadata['instrument_name'],
        'history': '',
    }
