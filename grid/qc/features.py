"""
grid.qc.features
================

"""


import numpy as np

from scipy import ndimage

from pyart.config import get_fillvalue, get_metadata

from . import compute_texture


def texture_fields(
        grid, fields=None, texture_window=(3, 3), texture_sample=5,
        fill_value=None, debug=False, verbose=False):
    """
    """

    # Parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse fields which we will compute the texture
    if fields is None:
        fields = grid.fields.keys()

    # Parse texture window parameters
    x_window, y_window = texture_window

    # Loop over all fields
    for field in fields:

        if verbose:
            print 'Computing texture field: {}'.format(field)

        _compute_texture(
            grid, field, x_window=x_window, y_window=y_window,
            min_sample=texture_sample, fill_value=fill_value,
            text_field=None, debug=debug)

    return


def significant_features(
        grid, field, structure=None, min_size=None, size_field=None,
        debug=False, verbose=False):
    """
    """

    # Parse field names
    if size_field is None:
        size_field = '{}_feature_size'.format(field)

    # Parse grid axes
    z_disp = grid.axes['z_disp']['data']

    for k, height in range(z_disp.size):

        print 'Here'

    return


def _compute_texture(grid, field, x_window=3, y_window=3, min_sample=5,
                     fill_value=None, text_field=None, debug=False):
    """
    """

    # Parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse field names
    if text_field is None:
        text_field = '{}_texture'.format(field)

    # Parse grid data
    data = grid.fields[field]['data']

    # Prepare data for ingest into Fortran wrapper
    data = np.ma.filled(data, fill_value)
    data = np.asfortranarray(data, dtype=np.float64)

    sample_size, texture = compute_texture.compute(
        data, x_window=x_window, y_window=y_window, fill_value=fill_value)

    # Mask grid points where the sample size used to compute the texture field
    # is too small
    if min_sample is not None:
        texture = np.ma.masked_where(
            sample_size < min_sample, texture, copy=False)

    # Mask invalid texture values and reset fill value
    np.ma.masked_values(texture, fill_value, atol=1.0e-8, copy=False)
    np.ma.masked_invalid(texture, copy=False)
    texture.set_fill_value(fill_value)

    # Add texture field to grid object
    field_dict = {
        'data': texture.astype(np.float32),
        'long_name': '{} texture'.format(grid.fields[field]['long_name']),
        '_FillValue': texture.fill_value,
        'units': grid.fields[field]['units'],
        'comment_1': ('Texture field is defined as the standard deviation '
                      'within a prescribed 2-D window'),
        'comment_2': '{} x {}'.format(x_window, y_window),
    }
    grid.fields[text_field] = field_dict

    return
