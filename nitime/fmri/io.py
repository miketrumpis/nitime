""" Input and output for fmri data files"""

try:
    from nibabel import load
except ImportError: 
        print "nibabel required for fmri I/O"

import nitime.timeseries as ts 
import nitime.analysis as tsa
import numpy as np

def time_series_from_file(nifti_file,coords,TR,normalize=None,average=False):
    """ Make a time series from a Analyze file, provided coordinates into the
            file 

    Parameters
    ----------

    nifti_file: string.

           The full path to the file from which the time-series is extracted 
     
    coords: ndarray
        x,y,z (inplane,inplane,slice) coordinates of the ROI from which the
        time-series is to be derived.
        
    TR: float, optional
        TR, if different from the one which can be extracted from the nifti
        file header

    normalize: Whether to normalize the activity in each voxel, defaults to
        None, in which case the original fMRI signal is used. Other options
        are: 'percent': the activity in each voxel is converted to percent
        change, relative to this scan. 'zscore': the activity is converted to a
        zscore relative to the mean and std in this voxel in this scan.

    average: bool, optional whether to average the time-series across the
           voxels in the ROI (assumed to be the first dimension). In which
           case, TS.data will be 1-d

    
    Returns
    -------

    time-series object

    Note
    ----

    Normalization occurs before averaging on a voxel-by-voxel basis, followed
    by the averaging. 

    """
    
    im = load(nifti_file)
    data = im.get_data()

    out_data = np.asarray(data[coords[0],coords[1],coords[2]])
        
    tseries = ts.TimeSeries(out_data,sampling_interval=TR)

    if normalize=='percent':
        tseries = tsa.NormalizationAnalyzer(tseries).percent_change
    elif normalize=='zscore':
        tseries = tsa.NormalizationAnalyzer(tseries).z_score

    if average:
        tseries.data = np.mean(tseries.data,0)
        
    return tseries


def nifti_from_time_series(volume,coords,time_series,nifti_path):
    """Makes a Nifti file out of a time_series object

    Parameters
    ----------

    volume: list (3-d, or 4-d)
        The total size of the nifti image to be created

    coords: 3*n_coords array
        The coords into which the time_series will be inserted. These need to
        be given in the order in which the time_series is organized

    time_series: a time-series object
       The time-series to be inserted into the file

    nifti_path: the full path to the file name which will be created
    
       """
    # XXX Implement! 
    raise NotImplementedError
