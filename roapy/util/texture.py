"""
grid.util.texture
=================

A submodule for computing texture fields from grid fields. A texture field is
defined as the standard deviation of a grid field within a prescribed 1-D or
2-D window.

"""


import numpy as np

from pyart.config import get_fillvalue, get_metadata

from . import _texture


def add_textures(grid, fields=None, window=(3, 3), min_sample=5,
                 fill_value=None, debug=False, verbose=False):
    """
    Add texture fields to grid fields dictionary.

    Parameters
    ----------
    grid : Grid
        Py-ART Grid containing specified fields.
    fields : str or list or tuple, optional
        Grid field(s) to compute texture field(s). If None, texture fields
        for all available grid fields will be computed and added.
    window : list or tuple, optional
        The 2-D (x, y) texture window used to compute texture fields.
    min_sample : int, optional
        Minimum sample size within texture window required to define a valid
        texture. Note that a minimum of 2 grid points are required to compute
        the texture field.
    fill_value : float, optional
        The value indicating missing or bad data in the grid field data. If
        None, the default value in the Py-ART configuration file is used.
    debug : bool, optional
        True to print debugging information, False to suppress.
    verbose : bool, optional
        True to print progress or identification information, False to
        suppress.

    """

    # Parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse the fields to compute textures from
    if fields is None:
        fields = grid.fields.keys()
    elif isinstance(fields, str):
        fields = [fields]
    else:
        fields = [field for field in fields if field in grid.fields]

    # Parse texture window parameters
    x_window, y_window = window

    for field in fields:

        if verbose:
            print 'Computing texture field: {}'.format(field)

        _add_texture(
            grid, field, x_window=x_window, y_window=y_window,
            min_sample=min_sample, fill_value=fill_value, text_field=None,
            debug=debug, verbose=verbose)

    return


def _add_texture(grid, field, x_window=3, y_window=3, min_sample=5,
                 fill_value=None, text_field=None, debug=False, verbose=False):
    """
    Compute the texture field (standard deviation) of the input grid field
    within a 1-D or 2-D window.

    Parameters
    ----------
    grid : Grid
        Py-ART Grid containing input field.
    field : str
        Input radar field used to compute the texture field.
    x_window : int, optional
        Number of x grid points in texture window.
    y_window : int, optional
        Number of y grid points in texture window.
    min_sample : int, optional
        Minimum sample size within texture window required to define a valid
        texture. Note that a minimum of 2 grid points are required to compute
        the texture field.
    fill_value : float, optional
        The value indicating missing or bad data in the grid field data. If
        None, the default value in the Py-ART configuration file is used.
    text_field : str, optional

    debug : bool, optional
        True to print debugging information, False to suppress.
    verbose : bool, optional
        True to print progress or identification information, False to
        suppress.

    """

    # Parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse field names
    if text_field is None:
        text_field = '{}_texture'.format(field)

    # Parse grid data
    data = grid.fields[field]['data'].copy()

    if debug:
        N = np.ma.count(data)
        print 'Sample size of data field: {}'.format(N)

    # Prepare data for ingest into Fortran wrapper
    data = np.ma.filled(data, fill_value)
    data = np.asfortranarray(data, dtype=np.float64)

    sigma, sample_size = _texture.compute_texture(
        data, x_window=x_window, y_window=y_window, fill_value=fill_value)

    # Mask grid points where sample size is insufficient
    if min_sample is not None:
        np.ma.masked_where(sample_size < min_sample, sigma, copy=False)
    np.ma.masked_values(sigma, fill_value, atol=1.0e-5, copy=False)
    np.ma.masked_invalid(sigma, copy=False)
    sigma.set_fill_value(fill_value)

    if debug:
        N = np.ma.count(sigma)
        print 'Sample size of texture field: {}'.format(N)

    # Create texture field dictionary
    sigma_dict = {
        'data': sigma,
        'units': '',
        '_FillValue': sigma.fill_value,
        }
    grid.add_field(text_field, sigma_dict, replace_existing=True)

    return
