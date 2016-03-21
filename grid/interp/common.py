"""
grid.interp.common
==================

Routines common to the interpolation module.

"""

import getpass
import platform
import numpy as np
from datetime import datetime

from pyart.config import get_metadata, get_fillvalue


def populate_field(data, inds, shape, field, weights=None, mask=None,
                   fill_value=None):
    """
    Create mapped radar field data dictionary.

    Parameters
    ----------
    data : ndarray
        Input radar data.
    inds : ndarray
        Indices corresponding to the k-nearest neighbours.
    shape : list-like
        Shape of analysis grid.
    field : str
        Field name.
    weights : ndarray, optional
        Distance-dependent weights applied to k-nearest neighbours. Use default
        None for nearest neighbor scheme. Must have same shape as inds.
    mask : ndarray, optional
        Masking will be applied where mask is True. Must have same shape as
        flattened grid.
    fill_value : float, optional
        Value indicating missing or bad data in input data. If None, default
        value in configuration file is used.

    Returns
    -------
    field_dict : dict
        Field dictionary containing data and metadata.

    """

    if fill_value is None:
        fill_value = get_fillvalue()

    if weights is None:
        fq = data[inds]
    else:
        fq = np.ma.average(data[inds], weights=weights, axis=1)

    fq = np.ma.masked_where(mask, fq, copy=False)
    fq.set_fill_value(fill_value)

    # Populate field dictionary
    field_dict = get_metadata(field)
    field_dict['data'] = fq.reshape(shape).astype(np.float32)
    if np.ma.is_masked(fq):
        field_dict['_FillValue'] = fq.fill_value

    return field_dict


def _populate_legacy_axes(radar, domain):
    """ Populate legacy grid axes data and metadata. """

    # Populate coordinate information
    x_disp = get_metadata('x')
    x_disp['data'] = domain.x.astype(np.float32)

    y_disp = get_metadata('y')
    y_disp['data'] = domain.y.astype(np.float32)

    z_disp = get_metadata('z')
    z_disp['data'] = domain.z.astype(np.float32)

    # Populate grid origin information
    alt = get_metadata('origin_altitude')
    alt['data'] = np.atleast_1d(domain.alt_0).astype(np.float32)

    lat = get_metadata('origin_latitude')
    lat['data'] = np.atleast_1d(domain.lat_0).astype(np.float32)

    lon = get_metadata('origin_longitude')
    lon['data'] = np.atleast_1d(domain.lon_0).astype(np.float32)

    # Populate grid time information
    time = get_metadata('grid_time')
    time['data'] = np.atleast_1d(radar.time['data'].min()).astype(np.float64)
    time['units'] = radar.time['units']

    time_start = get_metadata('grid_time_start')
    time_start['data'] = np.atleast_1d(
        radar.time['data'].min()).astype(np.float64)
    time_start['units'] = radar.time['units']

    time_end = get_metadata('grid_time_end')
    time_end['data'] = np.atleast_1d(
        radar.time['data'].max()).astype(np.float64)
    time_end['units'] = radar.time['units']

    return {
        'time': time,
        'time_start': time_start,
        'time_end': time_end,
        'x_disp': x_disp,
        'y_disp': y_disp,
        'z_disp': z_disp,
        'alt': alt,
        'lat': lat,
        'lon': lon,
        }


def _populate_metadata(radar, weight=None):
    """ Populate default grid metadata. """

    # Datastreams attribute (ARM standard)
    datastream_description = (
        'A string consisting of the datastream(s), datastream version(s), '
        'and datastream date (range).')

    # History attribute (ARM standard)
    history = 'created by user {} on {} at {}'.format(
        getpass.getuser(), platform.node(),
        datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))

    metadata = {
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
        'history': history,
        }

    if weight is not None:
        metadata['k_nearest_neighors'] = weight.k
        metadata['data_spacing'] = weight.data_spacing
        metadata['distance_weight_vanishes'] = weight.distance_weight_vanishes

    return metadata
