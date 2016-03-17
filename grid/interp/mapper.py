"""
grid.interp.mapper
==================

A module for mapping radar data to rectilinear grids. Space-partitioning data
structures known as kd-trees are used to efficiently search for k-nearest
neighbours.

"""

import os
import getpass
import platform
import numpy as np

from warnings import warn
from datetime import datetime
from scipy.spatial import cKDTree

from pyart.config import get_fillvalue, get_field_name, get_metadata
from pyart.core import Grid

from . import common
from ..util import transform
from ..core import Weight

# Necessary and/or potential future improvements to mapper submodule:
#
# * The time each radar measurement is recorded is reported for each ray, not
#   each gate, likely because it would be extreme overkill to record the
#   sampling time at each gate. Need to make sure the nearest neighbor time
#   field makes sense.


def grid_radar(radar, domain, weight=None, fields=None, gatefilter=None,
               toa=17000.0, max_range=None, legacy=False, fill_value=None,
               debug=False, verbose=False, **kwargs):
    """
    Map volumetric radar data to a rectilinear grid. This routine uses a k-d
    tree space-partitioning data structure for the efficient searching of the
    k-nearest neighbours.

    Parameters
    ----------
    radar : pyart.core.Radar
        Radar containing the fields to be mapped.
    domain : Domain
        Grid domain.
    weight : Weight, optional
        Weight defining the radar data objective analysis parameters and
        storage of available kd-tree information. Default uses an isotropic
        distance-dependent Barnes weight with a constant smoothing paramter.
    fields : list-like, optional
        Radar fields to be mapped. The None all available radar fields are
        mapped.
    gatefilter : pyart.filters.GateFilter, optional
        GateFilter indicating which radar gates should be excluded when
        mapping. The GateFilter is also used to determine the grid quality
        index.
    toa : float, optional
        Top of the atmosphere in meters. Radar gates above this altitude are
        ignored. Lower heights may increase processing time substantially but
        may also produce poor results if this value is similar or lower than
        the top of the grid domain.
    max_range : float, optional
        The maximum range of the radar in meters. Grid points not within this
        range from the radar are excluded.
    legacy : bool, optional
        True to return a legacy Py-ART Grid. Note that the legacy Grid object
        is planned for removal altogether in future Py-ART releases.
    debug : bool, optional
        True to print debugging information, False to suppress.
    verbose : bool, optional
        True to print relevant information, False to suppress.

    Return
    ------
    grid : pyart.core.Grid
        Grid containing the mapped volumetric radar data, grid coordinate
        information, and metadata.

    """

    # Parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse field names
    if kwargs.get('dist_field', None) is None:
        dist_field = get_field_name('nearest_neighbor_distance')
    if kwargs.get('weight_field', None) is None:
        weight_field = get_field_name('nearest_neighbor_weight')
    if kwargs.get('time_field', None) is None:
        time_field = get_field_name('nearest_neighbor_time')
    if kwargs.get('gqi_field', None) is None:
        gqi_field = get_field_name('grid_quality_index')

    # Parse fields to map
    if fields is None:
        fields = radar.fields.keys()
    elif isinstance(fields, str):
        fields = [fields]
    fields = [field for field in fields if field in radar.fields]

    # Parse radar data objective analysis weight
    if weight is None:
        weight = Weight(radar)

    # Parse maximum range
    if max_range is None:
        max_range = radar.range['data'].max()

    # Calculate radar offset relative to the analysis grid origin
    domain.radar_offset_from_origin(radar, debug=debug)

    # Compute Cartesian coordinates of radar gates relative to specified origin
    # Add reference gate locations and current gate locations to weight object
    # which will help determine if the kd-tree needs to be requeried or not
    z_g, y_g, x_g = transform.equivalent_earth_model(
        radar, offset=domain.radar_offset, debug=debug, verbose=verbose)
    weight._add_gate_reference([z_g, y_g, x_g], replace_existing=False)
    weight._add_gate_coordinates([z_g, y_g, x_g])

    if debug:
        print 'Number of radar gates before pruning: {}'.format(z_g.size)

    # Do not consider radar gates that are above the "top of the atmosphere"
    is_below_toa = z_g <= toa

    if debug:
        N = is_below_toa.sum()
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

    # Parse coordinates of analysis grid
    z_a, y_a, x_a = domain.z, domain.y, domain.x
    nz, ny, nx = domain.nz, domain.ny, domain.nx

    if debug:
        print 'Number of x grid points: {}'.format(nx)
        print 'Number of y grid points: {}'.format(ny)
        print 'Number of z grid points: {}'.format(nz)

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

        dists, inds = weight.query_tree(
            zip(z_a, y_a, x_a), store=True, debug=debug, verbose=verbose)

        # Compute distance-dependent weights
        wq = weight.compute_weights(weight.dists, store=True, verbose=verbose)

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
        N = is_bad_index.sum()
        print 'Number of invalid indices: {}'.format(N)

    # Grid points which are further than the specified maximum range away from
    # the radar should not contribute
    z_r, y_r, x_r = domain.radar_offset
    _range = np.sqrt((z_a - z_r)**2 + (y_a - y_r)**2 + (x_a - x_r)**2)
    is_far = _range > max_range

    if debug:
        N = is_far.sum()
        print('Number of analysis points too far from radar: {}'.format(N))

    # Populate grid fields
    map_fields = {}
    for field in fields:
        if verbose:
            print('Mapping radar field: {}'.format(field))

        map_fields[field] = common.populate_field(
            radar_data[field], inds, (nz, ny, nx), field, weights=wq,
            mask=is_far, fill_value=None)

    # Add grid quality index field
    if gatefilter is not None:

        # Compute distance-dependent weighted average of k-nearest neighbors
        # for included gates
        sqi = gatefilter.gate_included.flatten()[is_below_toa]
        gqi = np.average(sqi[inds], weights=wq, axis=1)
        gqi[is_far] = 0.0
        map_fields[gqi_field] = get_metadata(gqi_field)
        map_fields[gqi_field]['data'] = gqi.reshape(
            nz, ny, nx).astype(np.float32)

    # Add nearest neighbor distance field
    map_fields[dist_field] = get_metadata(dist_field)
    map_fields[dist_field]['data'] = dists[:,0].reshape(
        nz, ny, nx).astype(np.float32)

    # Add nearest neighbor weight field
    map_fields[weight_field] = get_metadata(weight_field)
    map_fields[weight_field]['data'] = wq[:,0].reshape(
        nz, ny, nx).astype(np.float32)

    # Add nearest neighbor time field
    time = radar.time['data'][:,np.newaxis].repeat(
        radar.ngates, axis=1).flatten()[is_below_toa][inds]
    map_fields[time_field] = get_metadata(time_field)
    map_fields[time_field]['data'] = time[:,0].reshape(
        nz, ny, nx).astype(np.float32)
    map_fields[time_field]['units'] = radar.time['units']

    # Populate grid metadata
    metadata = common._populate_metadata(radar, weight=weight)

    if legacy:
        axes = common._populate_legacy_axes(radar, domain)
        grid = Grid.from_legacy_parameters(map_fields, axes, metadata)
    else:
        grid = None

    return grid


def grid_radar_nearest_neighbour(
        radar, domain, fields=None, gatefilter=None, toa=17000.0,
        max_range=None, leafsize=10, legacy=False, fill_value=None,
        debug=False, verbose=False):
    """
    Map volumetric radar data to a rectilinear grid using nearest neighbour.

    Parameters
    ----------
    radar : pyart.core.Radar
        Radar containing the fields to be mapped.
    domain : Domain
        Grid domain.
    fields : list-like, optional
        Radar fields to be mapped. The None all available radar fields are
        mapped.
    gatefilter : pyart.filters.GateFilter, optional
        GateFilter indicating which radar gates should be excluded when
        mapping. The GateFilter is also used to determine the grid quality
        index.
    toa : float, optional
        Top of the atmosphere in meters. Radar gates above this altitude are
        ignored. Lower heights may increase processing time substantially but
        may also produce poor results if this value is similar or lower than
        the top of the grid domain.
    max_range : float, optional
        The maximum range of the radar in meters. Grid points not within this
        range from the radar are excluded.
    leafsize : int, optional
        The number of points at which the search algorithm switches over to
        brute-force.
    legacy : bool, optional
        True to return a legacy Py-ART Grid. Note that the legacy Grid object
        is planned for removal altogether in future Py-ART releases.
    debug : bool, optional
        True to print debugging information, False to suppress.
    verbose : bool, optional
        True to print relevant information, False to suppress.

    Return
    ------
    grid : pyart.core.Grid
        Grid containing the mapped volumetric radar data, coordinate
        information, and metadata.

    """

    # Parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse field names
    if kwargs.get('dist_field', None) is None:
        dist_field = get_field_name('nearest_neighbor_distance')
    if kwargs.get('weight_field', None) is None:
        weight_field = get_field_name('nearest_neighbor_weight')
    if kwargs.get('time_field', None) is None:
        time_field = get_field_name('nearest_neighbor_time')
    if kwargs.get('gqi_field', None) is None:
        gqi_field = get_field_name('grid_quality_index')

    # Parse maximum range
    if max_range is None:
        max_range = radar.range['data'].max()

    # Parse fields to map
    if fields is None:
        fields = radar.fields.keys()
    if isinstance(fields, str):
        fields = [fields]
    fields = [field for field in fields if field in radar.fields]

    # Calculate radar offset relative to the analysis grid origin
    domain.radar_offset_from_origin(radar, debug=debug)

    # Compute Cartesian coordinates of radar gates and apply origin offset
    z_g, y_g, x_g = transform.equivalent_earth_model(
        radar, offset=domain.radar_offset, debug=debug, verbose=verbose)

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
    tree_g = cKDTree(
        zip(z_g, y_g, x_g), leafsize=leafsize, compact_nodes=False,
        balanced_tree=False, copy_data=False)

    if debug:
        print 'tree.m = {}'.format(tree_g.m)
        print 'tree.n = {}'.format(tree_g.n)

    # Parse coordinates of analysis grid
    z_a, y_a, x_a = domain.z, domain.y, domain.x
    nz, ny, nx = domain.nz, domain.ny, domain.nx

    if debug:
        print 'Number of x grid points: {}'.format(nx)
        print 'Number of y grid points: {}'.format(ny)
        print 'Number of z grid points: {}'.format(nz)

    # Create analysis domain coordinates mesh
    z_a, y_a, x_a = np.meshgrid(z_a, y_a, x_a, indexing='ij')
    z_a, y_a, x_a = z_a.flatten(), y_a.flatten(), x_a.flatten()

    if debug:
        print 'Grid 1-D array shape: {}'.format(z_a.shape)

    # Query the radar gate k-d tree for the nearest analysis grid points
    # This is the step that consumes the most processing time
    if debug:
        print 'Querying k-d tree for the k-nearest analysis grid points'
    dists, inds = tree_g.query(
        zip(z_a, y_a, x_a), k=1, p=2.0, eps=0.0, distance_upper_bound=np.inf,
        n_jobs=1)

    if debug:
        print 'Distance array shape: {}'.format(dists.shape)
        print 'Minimum gate-grid distance: {:.2f} m'.format(dists.min())
        print 'Maximum gate-grid distance: {:.2f} m'.format(dists.max())
        print 'Index array shape: {}'.format(inds.shape)
        print 'Minimum index: {}'.format(inds.min())
        print 'Maximum index: {}'.format(inds.max())

    # Grid points which are further than the specified maximum range away from
    # the radar should not contribute
    z_r, y_r, x_r = domain.radar_offset
    _range = np.sqrt((z_a - z_r)**2 + (y_a - y_r)**2 + (x_a - x_r)**2)
    is_far = _range > max_range

    if debug:
        n = is_far.sum()
        print 'Number of analysis points too far from radar: {}'.format(n)

    # Populate grid fields
    map_fields = {}
    for field in fields:
        if debug:
            print 'Mapping radar field: {}'.format(field)

        map_fields[field] = common.populate_field(
            radar_data[field], inds, (nz, ny, nx), field, weights=None,
            mask=is_far, fill_value=None)

    # Add grid quality index field
    if gatefilter is not None:

        gqi = gatefilter.gate_included.flatten()[is_below_toa][inds]
        gqi[is_far] = 0.0
        map_fields[gqi_field] = get_metadata(gqi_field)
        map_fields[gqi_field]['data'] = gqi.reshape(
            nz, ny, nx).astype(np.float32)

    # Add nearest neighbor distance field
    map_fields[dist_field] = get_metadata(dist_field)
    map_fields[dist_field]['data'] = dists.reshape(
        nz, ny, nx).astype(np.float32)

    # Add nearest neighbor time field
    time = radar.time['data'][:,np.newaxis].repeat(
        radar.ngates, axis=1).flatten()[is_below_toa][inds]
    map_fields[time_field] = get_metadata(time_field)
    map_fields[time_field]['data'] = time.reshape(
        nz, ny, nx).astype(np.float32)
    map_fields[time_field]['units'] = radar.time['units']

    # Populate grid metadata
    metadata = common._populate_metadata(radar, weight=None)

    if legacy:
        axes = common._populate_legacy_axes(radar, domain)
        grid = Grid.from_legacy_parameters(map_fields, axes, metadata)
    else:
        grid = None

    return grid
