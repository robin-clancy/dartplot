import cartopy.crs as ccrs
import datetime
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr

from glob import glob
from matplotlib import colormaps
    

def load_truth(archive, truth, time, lev=False):
    """
    Loads atmosphere history file from truth case into xarray DataArray

    Inputs:
    archive (str) --> path to directory containing history files e.g. f"/glade/derecho/scratch/{USER}/archive"
    truth (str) --> name of truth case
    time (str) --> time of hist file to be loaded e.g. "2020-08-08-21600"
    lev (int) --> model vertical level to be plotted 

    Returns:
    ds_truth --> xarray DataArray
    """
    truth_file = os.path.join(archive, truth, "atm", "hist", f"{truth}.cam.h0.{time}.nc")
    ds_truth = xr.open_mfdataset(truth_file).isel(time=0)

    if lev:
        ds_truth = ds_truth.isel(lev=lev)
    return ds_truth


def load_forecast(archive, exp, time, lev=False, sd=False):
    """
    Loads forecast mean or forecast standard deviation from DART experiment into xarray DataArray

    Forecast files can be zipped as .gz or unzipped

    Inputs:
    archive (str) --> path to directory containing forecast files e.g. f"/glade/derecho/scratch/{USER}/archive"
    exp (str) --> name of experiment case
    time (str) --> time of forecast file to be loaded e.g. "2020-08-08-21600"
    lev (int) --> model vertical level to be plotted 
    sd (boolean) --> set to True to load forecast standard deviation instead of forecast

    Returns:
    ds_forecast --> xarray DataArray
    """
    if sd:
        forecast_file = os.path.join(archive, exp, "esp", "hist", f"{exp}.dart.e.cam_forecast_sd.{time}.nc")
    else:
        forecast_file = os.path.join(archive, exp, "esp", "hist", f"{exp}.dart.e.cam_forecast_mean.{time}.nc")

    try:
        ds_forecast = xr.open_mfdataset(forecast_file).isel(time=0)
    except:
        ds_forecast = xr.open_mfdataset(forecast_file + ".gz").isel(time=0)

    if lev:
        ds_forecast = ds_forecast.isel(lev=lev)
    return ds_forecast


def load_output(archive, exp, time, lev=False, sd=False):
    """
    Loads output mean or output standard deviation from DART experiment into xarray DataArray

    Output files can be zipped as .gz or unzipped

    Inputs:
    archive (str) --> path to directory containing output files e.g. f"/glade/derecho/scratch/{USER}/archive"
    exp (str) --> name of experiment case
    time (str) --> time of output file to be loaded e.g. "2020-08-08-21600"
    lev (int) --> model vertical level to be plotted 
    sd (boolean) --> set to True to load output standard deviation instead of forecast

    Returns:
    ds_output --> xarray DataArray
    """
    if sd:
        output_file = os.path.join(archive, exp, "esp", "hist", f"{exp}.dart.i.cam_output_sd.{time}.nc")
    else:
        output_file = os.path.join(archive, exp, "esp", "hist", f"{exp}.dart.i.cam_output_mean.{time}.nc")

    try:
        ds_output = xr.open_dataset(output_file).isel(time=0)
    except:
        ds_output = xr.open_dataset(output_file + ".gz").isel(time=0)

    if lev:
        ds_output = ds_output.isel(lev=lev)
    return ds_output


def load_increment(ds_output, ds_forecast):
    """
    Returns increment between forecast and output for a DART experiment as an xarray DataArray

    Inputs:
    ds_output (xarray DataArray) --> see state.load_output
    ds_forecast (xarray DataArray) --> see state.load_forecast

    Returns:
    ds_increment --> xarray DataArray
    """
    ds_increment = ds_output - ds_forecast
    return ds_increment


def make_map(data, lon, lat, clim_even=False, cmax=False, cmap='viridis'):
    """
    Plots map data on CAM-SE grid
    
    The data is transformed to the projection using Cartopy's `transform_points` method.

    The plot is made by triangulation of the points, producing output very similar to `pcolormesh`,
    but with triangles instead of rectangles used to make the image.
   
    Mostly taken from some forum post somewhere with some minor adaptations made.

    Inputs:
    data (xarray DataArray) --> for one variable e.g. ds_truth['T'] for temperature
    lat, lon (xaray DataArray) --> 1D (ncol) of lons and another for lats
    clim_even (boolean) --> if True centers colormap around 0
    cmax (int) --> sets colormap maximum value
    cmap (str) --> name of matplotlib colormap
    
    Outputs:
    fig, ax, img --> figure, axis and image objects
    """
    dataproj = ccrs.PlateCarree() # assumes data is lat/lon
    #plotproj = ccrs.Mollweide()   # output projection
    plotproj = ccrs.PlateCarree()   # output projection
    # set up figure / axes object, set to be global, add coastlines
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection':plotproj})
    ax.set_global()
    ax.coastlines(linewidth=0.2)
    # this figures out the transformation between (lon,lat) and the specified projection
    tcoords = plotproj.transform_points(dataproj, lon, lat) # working with the projection
    xi=tcoords[:,0] != np.inf  # there can be bad points set to infinity, but we'll ignore them
    # print(f"{xi.shape = } --- Number of False: {np.count_nonzero(~xi)}")
    tc=tcoords[xi,:]
    datai=data[xi]  # convert to numpy array, then subset
    # Use tripcolor --> triangluates the data to make the plot
    # rasterized=True reduces the file size (necessary for high-resolution for reasonable file size)
    # keep output as "img" to make specifying colorbar easy
    img = ax.tripcolor(tc[:,0], tc[:,1], datai, cmap=cmap, shading='gouraud', rasterized=True)
    if clim_even:
        if cmax:
            img.set_clim((-cmax, cmax))
        else:
            img.set_clim((-max(np.abs(img.get_clim())), max(np.abs(img.get_clim()))))
    cbar = fig.colorbar(img, ax=ax, shrink=0.4)
    return fig, ax, img


def get_times(archive, exp, stage, value, start_ind=False, end_ind=False, step=False):
    """
    Function to get available times based on file names

    Inputs:
    archive (str) --> path to directory containing output files e.g. f"/glade/derecho/scratch/{USER}/archive"
    exp (str) --> name of experiment case
    stage (str) --> "forecast" or "output"
    value (str) --> value = "mean" or "sd" for standard deviation
    Optional Inputs:
    start_ind, end_int, step (int) --> start and end indices of file list to select. use step for time steps other than 1.

    Returns:
    times (list) --> list of times
    ntimes (int) --> length of the list of times
    """
    files = glob(f"{archive}/{exp}/esp/hist/*{stage}_{value}*")
    files.sort()
    times = [x.split(f'_{value}.')[1].split('.nc')[0] for x in files]
    if start_ind:
        times = times[start_ind:end_ind]
    if step:
        times = times[0:-1:step]

    ntimes = len(times)
    return times, ntimes


def zmean_unweighted(var, lat):
    """
    Calculates zonal mean of a variable
    Weighting is equal for all grid cells: not area weighted!
    Needs editing if you want to use a variable on a latxlon grid
    
    Inputs:
    var (xarray DataArray) --> variable to calculate zonal mean of, dims=ncol currently
    lat (xarray DataArray) --> latitude values associated with var, dims=ncol currently

    Returns
    zmean (numpy array) --> zonal mean values
    zlat (numpy array) --> centers of latitude values
    """
    lat = lat.values
    zlat = np.arange(-90.5, 89.5, 1)
    zmean = []
    for y in zlat:
        zmean.append(np.nanmean(var.values[np.logical_and(lat > y, lat < y + 1)]))
    zmean = np.array(zmean)
    zlat = zlat + 0.5 #moving from edges to center of grid
    return zmean, zlat


def plot_zmean(zmean, zlat, ax, xlabel='', title='', color=False):
    """
    Line plot of zonal mean
    
    Inputs:
    zmean (numpy array) --> zonal mean values
    zlat (numpy array)--> centers of latitude values
    ax --> matplotlib axis to plot on
    xlabel (str) --> x axis label
    title (str)
    color (str) --> matplotlib colormap

    Returns:
    ax  --> matplotlib axis
    """
    if color is not False:
        ax.plot(zmean, zlat, color=color)
    else:    
        ax.plot(zmean, zlat)

    if (0 > ax.get_xlim()[0]) & (0 < ax.get_xlim()[1]):
        ax.axvline(0, color=[0, 0, 0], alpha=0.2)
    
    ax.set_title(title, fontsize=16)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    
    ax.set_ylabel('Latitude', fontsize=16)
    ax.set_ylim([-90, 90])
    ax.set_yticks(np.arange(-90, 90+1, 30))
    ax.tick_params(axis='y', labelsize=12)
    
    ax.grid(color = [0.5, 0.5, 0.5], linestyle = '--', linewidth = 0.4)
    return ax


def calculate_zmean_timeseries(archive, truth, exp, var, lev, value, abs,
                               start_ind=False, end_ind=False, step=False):
    """
    Wrapper for all the zonal mean calculating functions
    
    e.g. inputs:
    archive =  "/glade/derecho/scratch/chennuo/archive"
    truth = "CESM2_2_BHIST_ne0ARCTICne30x4_g17_derecho_2560pes_spinup_allvars_from_20200701_2mos.004"
    exp = "FHIST_exp.1A_0001"
    var = 'T'
    lev = 25
    abs = True if want magnitude, False if want bias - Decides if using absolute error and increment

    value_options:
    "forecast_error", "output_error", "increment", "improvement",
    "forecast_sd", "output_sd", "ratio_sd"
    """
    
    if value in ["forecast_error"]:
        zmeans, zlat, times = calculate_zmean_timeseries_forecast_error(archive, truth, exp, var, lev, abs,
                                                                        start_ind=start_ind, end_ind=end_ind, step=step)
    elif value in ["output_error"]:
        zmeans, zlat, times = calculate_zmean_timeseries_output_error(archive, truth, exp, var, lev, abs,
                                                                      start_ind=start_ind, end_ind=end_ind, step=step)    
    elif value in ["increment"]:
        zmeans, zlat, times = calculate_zmean_timeseries_increment(archive, truth, exp, var, lev, abs,
                                                                   start_ind=start_ind, end_ind=end_ind, step=step)
    elif value in ["improvement"]:
        zmeans, zlat, times = calculate_zmean_timeseries_improvement(archive, truth, exp, var, lev, abs,
                                                                     start_ind=start_ind, end_ind=end_ind, step=step)
    elif value in ["forecast_sd"]:
        zmeans, zlat, times = calculate_zmean_timeseries_forecast_sd(archive, truth, exp, var, lev, abs,
                                                                     start_ind=start_ind, end_ind=end_ind, step=step)
    elif value in ["output_sd"]:
        zmeans, zlat, times = calculate_zmean_timeseries_output_sd(archive, truth, exp, var, lev, abs,
                                                                   start_ind=start_ind, end_ind=end_ind, step=step)
    elif value in ["ratio_sd"]:
        zmeans, zlat, times = calculate_zmean_timeseries_ratio_sd(archive, truth, exp, var, lev, abs,
                                                                  start_ind=start_ind, end_ind=end_ind, step=step)
        
    return zmeans, zlat, times


def calculate_zmean_timeseries_forecast_error(archive, truth, exp, var, lev, abs,
                                              start_ind=False, end_ind=False, step=False):
    zmeans = []
    times, ntimes = get_times(archive, exp, stage="forecast", value="mean",
                              start_ind=start_ind, end_ind=end_ind, step=step)
    lat = load_forecast(archive, exp, times[0], lev=lev)['lat']
    for i, time in enumerate(times[0:ntimes]):
        if abs:
            zmean, zlat = zmean_unweighted(np.abs(load_forecast(archive, exp, time, lev=lev)[var] - 
                                           load_truth(archive, truth, time, lev=lev)[var]), lat)
        else:
            zmean, zlat = zmean_unweighted(load_forecast(archive, exp, time, lev=lev)[var] - 
                                           load_truth(archive, truth, time, lev=lev)[var], lat)
        zmeans.append(zmean)
    zmeans = np.vstack(zmeans).T
    return zmeans, zlat, times


def calculate_zmean_timeseries_output_error(archive, truth, exp, var, lev, abs,
                                            start_ind=False, end_ind=False, step=False):
    zmeans = []
    times, ntimes = get_times(archive, exp, stage="output", value="mean",
                              start_ind=start_ind, end_ind=end_ind, step=step)
    lat = load_output(archive, exp, times[0], lev=lev)['lat']
    for i, time in enumerate(times[0:ntimes]):
        if abs:
            zmean, zlat = zmean_unweighted(np.abs(load_output(archive, exp, time, lev=lev)[var] - 
                                           load_truth(archive, truth, time, lev=lev)[var]), lat)
        else:
            zmean, zlat = zmean_unweighted(load_output(archive, exp, time, lev=lev)[var] - 
                                           load_truth(archive, truth, time, lev=lev)[var], lat)
        zmeans.append(zmean)
    zmeans = np.vstack(zmeans).T
    return zmeans, zlat, times



def calculate_zmean_timeseries_increment(archive, truth, exp, var, lev, abs,
                                         start_ind=False, end_ind=False, step=False):
    zmeans = []
    times, ntimes = get_times(archive, exp, stage="output", value="mean",
                              start_ind=start_ind, end_ind=end_ind, step=step)
    lat = load_forecast(archive, exp, times[0], lev=lev)['lat']
    for i, time in enumerate(times[0:ntimes]):
        if abs:
            zmean, zlat = zmean_unweighted(np.abs(load_increment(load_output(archive, exp, time, lev=lev)[var],
                                                                 load_forecast(archive, exp, time, lev=lev))[var]), lat)
        else:
            zmean, zlat = zmean_unweighted(load_increment(load_output(archive, exp, time, lev=lev)[var],
                                                          load_forecast(archive, exp, time, lev=lev)[var]), lat)
        zmeans.append(zmean)
    zmeans = np.vstack(zmeans).T
    return zmeans, zlat, times


def calculate_zmean_timeseries_improvement(archive, truth, exp, var, lev, abs,
                                           start_ind=False, end_ind=False, step=False):
    zmeans = []
    times, ntimes = get_times(archive, exp, stage="output", value="mean",
                              start_ind=start_ind, end_ind=end_ind, step=step)
    lat = load_forecast(archive, exp, times[0], lev=lev)['lat']
    for i, time in enumerate(times[0:ntimes]):
        zmean, zlat = zmean_unweighted(np.abs(load_forecast(archive, exp, time, lev=lev)[var] - 
                                       load_truth(archive, truth, time, lev=lev)[var]) - 
                                       np.abs(load_output(archive, exp, time, lev=lev)[var] - 
                                       load_truth(archive, truth, time, lev=lev)[var]), lat)
        zmeans.append(zmean)
    zmeans = np.vstack(zmeans).T
    return zmeans, zlat, times


def calculate_zmean_timeseries_forecast_sd(archive, truth, exp, var, lev, abs,
                                           start_ind=False, end_ind=False, step=False):
    zmeans = []
    times, ntimes = get_times(archive, exp, stage="forecast", value="sd",
                              start_ind=start_ind, end_ind=end_ind, step=step)
    lat = load_forecast(archive, exp, times[0], lev=lev, sd=True)['lat']
    for i, time in enumerate(times[0:ntimes]):
        zmean, zlat = zmean_unweighted(load_forecast(archive, exp, time, lev=lev, sd=True)[var], lat)
        zmeans.append(zmean)
    zmeans = np.vstack(zmeans).T
    return zmeans, zlat, times


def calculate_zmean_timeseries_output_sd(archive, truth, exp, var, lev, abs,
                                         start_ind=False, end_ind=False, step=False):
    zmeans = []
    times, ntimes = get_times(archive, exp, stage="output", value="sd",
                              start_ind=start_ind, end_ind=end_ind, step=step)
    lat = load_output(archive, exp, times[0], lev=lev, sd=True)['lat']
    for i, time in enumerate(times[0:ntimes]):
        zmean, zlat = zmean_unweighted(load_output(archive, exp, time, lev=lev, sd=True)[var], lat)
        zmeans.append(zmean)
    zmeans = np.vstack(zmeans).T
    return zmeans, zlat, times


def calculate_zmean_timeseries_ratio_sd(archive, truth, exp, var, lev, abs,
                                         start_ind=False, end_ind=False, step=False):
    zmeans = []
    times, ntimes = get_times(archive, exp, stage="forecast", value="sd",
                              start_ind=start_ind, end_ind=end_ind, step=step)
    lat = load_forecast(archive, exp, times[0], lev=lev, sd=True)['lat']
    for i, time in enumerate(times[0:ntimes]):
        zmean, zlat = zmean_unweighted(load_output(archive, exp, time, lev=lev, sd=True)[var] / 
                                       load_forecast(archive, exp, time, lev=lev, sd=True)[var], lat)
        zmeans.append(zmean)
    zmeans = np.vstack(zmeans).T
    return zmeans, zlat, times


def plot_zmean_timeseries(zmeans, zlat, xlabel='', title='', cmap='magma', cmapscale=0.8):
    """
    Lines plot of timeseries of zonal means
    
    Inputs:
    zmeans (numpy array) --> time series of zonal mean values
    zlat (numpy array)--> centers of latitude values
    xlabel (str) --> x axis label
    title (str)
    color (str) --> matplotlib colormap
    cmapscale (float) --> value to scale matplotlib colormap values by (e.g. 0.8 darkens 20%)

    Returns:
    fig, ax  --> matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    n = zmeans.shape[1]
    rgba_color = get_rgba_color(cmap, n)  * cmapscale
    for i in np.arange(0, n):
        ax.plot(zmeans[:,i], zlat, color=rgba_color[i,:])

    if (0 > ax.get_xlim()[0]) & (0 < ax.get_xlim()[1]):
        ax.axvline(0, color=[0, 0, 0], alpha=0.2)
    
    ax.set_title(title, fontsize=16)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    
    ax.set_ylabel('Latitude', fontsize=16)
    ax.set_ylim([-90, 90])
    ax.set_yticks(np.arange(-90, 90+1, 30))
    ax.tick_params(axis='y', labelsize=12)
    
    ax.grid(color = [0.5, 0.5, 0.5], linestyle = '--', linewidth = 0.4)
    return fig, ax


def pcolor_zmean_timeseries(zmeans, zlat, times, label='', title='', clim_even=False, cmap='magma'):
    """
    Lines plot of timeseries of zonal means
    
    Inputs:
    zmeans (numpy array) --> time series of zonal mean values
    zlat (numpy array)--> centers of latitude values
    times (list) --> list of times for each zonal mean
    label (str) --> colorbar label
    title (str)
    clim_even (boolean) --> if True centers colormap around 0
    cmap (str) --> matplotlib colormap
    
    Returns:
    fig, ax  --> matplotlib figure and axis
    """
    
    dates = [datetime.datetime.strptime('-'.join(x.split('-')[0:3]), "%Y-%m-%d") + 
             datetime.timedelta(seconds = int(x.split('-')[3])) for x in times]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    img = plt.pcolor(dates, zlat, zmeans, cmap=cmap)

    if clim_even:
        img.set_clim((-max(np.abs(img.get_clim())), max(np.abs(img.get_clim()))))
    
    cb = plt.colorbar()
    cb.set_label(label=label, size=14)
    cb.get_ticks
    cb.ax.tick_params(labelsize=12)
    
    ax.set_title(title, fontsize=16)
    
    ax.set_xlabel('Date', fontsize=14);
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%-HZ'))
    ax.tick_params(axis='x', labelsize=12)
    
    ax.set_ylabel('Latitude', fontsize=14);
    ax.set_ylim([-90, 90])
    ax.set_yticks(np.arange(-90, 90+1, 30))
    ax.tick_params(axis='y', labelsize=12)
    
    ax.grid(color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.4)

    return fig, ax

def calculate_vmean_timeseries(archive, truth, exp, var, value, abs,
                               start_ind=False, end_ind=False, step=False):
    """
    Wrapper for all the vertical profile calculating functions
    
    e.g. inputs:
    archive =  "/glade/derecho/scratch/chennuo/archive"
    truth = "CESM2_2_BHIST_ne0ARCTICne30x4_g17_derecho_2560pes_spinup_allvars_from_20200701_2mos.004"
    exp = "FHIST_exp.1A_0001"
    var = 'T'
    abs = True if want magnitude, False if want bias - Decides if using absolute error and increment

    value_options:
    "forecast_error", "output_error", "increment", "improvement",
    "forecast_sd", "output_sd", "ratio_sd"
    """
    
    if value in ["forecast_error"]:
        vmeans, levs, times = calculate_vmean_timeseries_forecast_error(archive, truth, exp, var, abs,
                                                                        start_ind=start_ind, end_ind=end_ind, step=step)
    elif value in ["output_error"]:
        vmeans, levs, times = calculate_vmean_timeseries_output_error(archive, truth, exp, var, abs,
                                                                      start_ind=start_ind, end_ind=end_ind, step=step)    
    elif value in ["increment"]:
        vmeans, levs, times = calculate_vmean_timeseries_increment(archive, truth, exp, var, abs,
                                                                   start_ind=start_ind, end_ind=end_ind, step=step)
    elif value in ["improvement"]:
        vmeans, levs, times = calculate_vmean_timeseries_improvement(archive, truth, exp, var, abs,
                                                                     start_ind=start_ind, end_ind=end_ind, step=step)
    elif value in ["forecast_sd"]:
        vmeans, levs, times = calculate_vmean_timeseries_forecast_sd(archive, truth, exp, var, abs,
                                                                     start_ind=start_ind, end_ind=end_ind, step=step)
    elif value in ["output_sd"]:
        vmeans, levs, times = calculate_vmean_timeseries_output_sd(archive, truth, exp, var, abs,
                                                                   start_ind=start_ind, end_ind=end_ind, step=step)
    elif value in ["ratio_sd"]:
        vmeans, levs, times = calculate_vmean_timeseries_ratio_sd(archive, truth, exp, var, abs,
                                                                  start_ind=start_ind, end_ind=end_ind, step=step)
        
    return vmeans, levs, times


def calculate_vmean_timeseries_forecast_error(archive, truth, exp, var, abs,
                                              start_ind=False, end_ind=False, step=False):
    vmeans = []
    times, ntimes = get_times(archive, exp, stage="forecast", value="mean",
                                    start_ind=start_ind, end_ind=end_ind, step=step)
    for i, time in enumerate(times[0:ntimes]):
        ds_forecast = load_forecast(archive, exp, time)[var]
        ds_truth = load_truth(archive, truth, time)[var]
        
        # Need this step as lev isn't identical over both
        ds_forecast['lev'] = ds_truth['lev']
        
        if abs:
            vmean = np.abs(ds_forecast - ds_truth).mean(dim="ncol")
        else:
            vmean = (ds_forecast - ds_truth).mean(dim="ncol")
            
        vmeans.append(vmean)
    vmeans = np.vstack(vmeans).T
    return vmeans, ds_forecast['lev'], times


def calculate_vmean_timeseries_output_error(archive, truth, exp, var, abs,
                                            start_ind=False, end_ind=False, step=False):
    vmeans = []
    times, ntimes = get_times(archive, exp, stage="output", value="mean",
                                    start_ind=start_ind, end_ind=end_ind, step=step)
    for i, time in enumerate(times[0:ntimes]):
        ds_output = load_output(archive, exp, time)[var]
        ds_truth = load_truth(archive, truth, time)[var]
        
        # Need this step as lev isn't identical over both
        ds_output['lev'] = ds_truth['lev']
        
        if abs:
            vmean = np.abs(ds_forecast - ds_truth).mean(dim="ncol")
        else:
            vmean = (ds_forecast - ds_truth).mean(dim="ncol")
            
        vmeans.append(vmean)
    vmeans = np.vstack(vmeans).T
    return vmeans, ds_output['lev'], times


def calculate_vmean_timeseries_increment(archive, truth, exp, var, abs,
                                         start_ind=False, end_ind=False, step=False):
    vmeans = []
    times, ntimes = get_times(archive, exp, stage="output", value="mean",
                                    start_ind=start_ind, end_ind=end_ind, step=step)
    for i, time in enumerate(times[0:ntimes]):
        ds_output = load_output(archive, exp, time)[var]
        ds_forecast = load_forecast(archive, exp, time)[var]
        
        # Need this step as lev isn't identical over both
        ds_output['lev'] = ds_forecast['lev']
        
        if abs:
            vmean = np.abs(load_increment(ds_output, ds_forecast)).mean(dim="ncol")
        else:
            vmean = load_increment(ds_output, ds_forecast).mean(dim="ncol")
            
        vmeans.append(vmean)
    vmeans = np.vstack(vmeans).T
    return vmeans, ds_output['lev'], times


def calculate_vmean_timeseries_improvement(archive, truth, exp, var, abs,
                                           start_ind=False, end_ind=False, step=False):
    vmeans = []
    times, ntimes = get_times(archive, exp, stage="output", value="mean",
                                    start_ind=start_ind, end_ind=end_ind, step=step)
    for i, time in enumerate(times[0:ntimes]):
        ds_output = load_output(archive, exp, time)[var]
        ds_forecast = load_forecast(archive, exp, time)[var]
        ds_truth = load_truth(archive, truth, time)[var]
        
        # Need this step as lev isn't identical over both
        ds_output['lev'] = ds_truth['lev']
        ds_forecast['lev'] = ds_truth['lev']
        
        vmean = (np.abs(ds_forecast - ds_truth) - np.abs(ds_output - ds_truth)).mean(dim="ncol")
            
        vmeans.append(vmean)
    vmeans = np.vstack(vmeans).T
    return vmeans, ds_output['lev'], times
    

def calculate_vmean_timeseries_forecast_sd(archive, truth, exp, var, abs,
                                           start_ind=False, end_ind=False, step=False):
    vmeans = []
    times, ntimes = get_times(archive, exp, stage="forecast", value="sd",
                                    start_ind=start_ind, end_ind=end_ind, step=step)
    for i, time in enumerate(times[0:ntimes]):
        ds_forecast = load_forecast(archive, exp, time, sd=True)[var]
        vmean = ds_forecast.mean(dim="ncol")
        vmeans.append(vmean)
    vmeans = np.vstack(vmeans).T
    return vmeans, ds_forecast['lev'], times


def calculate_vmean_timeseries_output_sd(archive, truth, exp, var, abs,
                                         start_ind=False, end_ind=False, step=False):
    vmeans = []
    times, ntimes = get_times(archive, exp, stage="output", value="sd",
                                    start_ind=start_ind, end_ind=end_ind, step=step)
    for i, time in enumerate(times[0:ntimes]):
        ds_output = load_output(archive, exp, time, sd=True)[var]
        vmean = ds_output.mean(dim="ncol")
        vmeans.append(vmean)
    vmeans = np.vstack(vmeans).T
    return vmeans, ds_output['lev'], times


def calculate_vmean_timeseries_ratio_sd(archive, truth, exp, var, abs,
                                        start_ind=False, end_ind=False, step=False):
    vmeans = []
    times, ntimes = get_times(archive, exp, stage="output", value="sd",
                                    start_ind=start_ind, end_ind=end_ind, step=step)
    for i, time in enumerate(times[0:ntimes]):
        ds_output = load_output(archive, exp, time, sd=True)[var]
        ds_forecast = load_forecast(archive, exp, time, sd=True)[var]
        vmean = (ds_output / ds_forecast).mean(dim="ncol")
        vmeans.append(vmean)
    vmeans = np.vstack(vmeans).T
    return vmeans, ds_output['lev'], times


def pcolor_vmean_timeseries(vmeans, times, levs, label='', title='', clim_even=False, cmap='magma'):
    """
    Lines plot of timeseries of vertical profiles
    
    Inputs:
    zmeans (numpy array) --> time series of zonal mean values
    zlat (numpy array)--> centers of latitude values
    levs (list) --> list of levels to be plotted
    label (str) --> colorbar label
    title (str)
    clim_even (boolean) --> if True centers colormap around 0
    cmap (str) --> matplotlib colormap
    
    Returns:
    fig, ax  --> matplotlib figure and axis
    """
    dates = [datetime.datetime.strptime('-'.join(x.split('-')[0:3]), "%Y-%m-%d") + 
             datetime.timedelta(seconds = int(x.split('-')[3])) for x in times]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    img = plt.pcolor(dates, levs, vmeans, cmap=cmap)

    if clim_even:
        img.set_clim((-max(np.abs(img.get_clim())), max(np.abs(img.get_clim()))))
    
    cb = plt.colorbar()
    cb.set_label(label=label, size=14)
    cb.get_ticks
    cb.ax.tick_params(labelsize=12)

    ax.invert_yaxis()
    
    ax.set_title(title, fontsize=16)
    
    ax.set_xlabel('Date', fontsize=14);
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%-HZ'))
    ax.tick_params(axis='x', labelsize=12)
    
    ax.set_ylabel('Level', fontsize=14);
    #ax.set_ylim([-90, 90])
    #ax.set_yticks(np.arange(-90, 90+1, 30))
    ax.tick_params(axis='y', labelsize=12)
    
    ax.grid(color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.4)

    return fig, ax


def get_rgba_color(color, n=False):
    """
    Gets list of colors to use for plotting from various potential inputs
    
    e.g.:
    get_rgba_color('r')
    get_rgba_color(['r', 'xkcd:blue', 'green'])
    get_rgba_color('magma', 5)
    get_rgba_color(ax1, [1,3])
    
    Inupts:
    color (string "red", short string "r",
           list of strings/short strings ['r', 'xkcd:blue', 'green'],
           matplotlib cmap name as string'turbo',
           axis ax1 with colored lines already plotted that need matching,
           np.ndarray rgba array([1.4620e-03, 4.6600e-04, 1.3866e-02, 1.0000e+00])
          )
    n (integer, optional) --> for use with colormap, determines number of colors to select from colormap
    n (integer, list, optional) --> for use with np.array or axis to select which index/indices of existing color list to use 
    
    Returns:
    rgba_color (numpy.ndarray) --> n x 4 (r, g, b, a) array describing colors list for use in plots
    """
    
    if isinstance(color, np.ndarray):
        if n is not False:
            if not isinstance(n, list):
                n = [n]
            rgba_color = color[n]
        else:
            rgba_color = color

    elif isinstance(color, mpl.axes._axes.Axes):
        color = np.array([x.get_c() for x in color.get_lines()])
        if n is not False:
            if not isinstance(n, list):
                n = [n]
            rgba_color = color[n]
        else:
            rgba_color = color
            
    elif color in list(colormaps):
        cmap = colormaps[color]
        rgba_color = cmap(np.linspace(0, 1, n))
        
    else:
        rgba_color = mcolors.to_rgba_array(color)
        
    return rgba_color