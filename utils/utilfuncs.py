import numpy as np
import xarray as xr

# Function to convert longitude
def reorient_netCDF(fp):
    """
    Function to orient and save netcdf wrt -180,180 longitude.
    :param fp: dataarray to be reoriented
    """
    f = fp
    if np.max(f.coords['lon'])>180:
        new_lon = [-360.00 + num if num > 180 else num for num in f.coords['lon'].values]
        f = f.assign_coords({'lon':new_lon})
        f.assign_coords(lon=(np.mod(f.lon + 180, 360) - 180))
        f = f.sortby(f.coords['lon'])
    return f

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
	FUNCTION COPIED DIRECTLY FROM https://stackoverflow.com/a/29677616/12133280
    """

    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


