import itertools
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import xarray as xr

from matplotlib import colormaps
from netCDF4 import Dataset


def list_vname_options(fname):
    fh = Dataset(fname, mode='r')
    variable_options = fh.variables.keys()
    fh.close()
    return variable_options

def list_stat_options(fname):
    fh = Dataset(fname, mode='r')
    cmd = fh.variables["CopyMetaData"]
    cmd_list = [str(netCDF4.chartostring(cmd[x,:].data)).strip() for x in range(0, cmd.shape[0])]
    fh.close()
    return cmd_list

def list_plevel_options(fname):
    # Could rewrite this in xarray so it lists pressure level options for a specified variable
    fh = Dataset(fname, mode='r')
    plevel_options = [float(x) for x in fh.variables["plevel"]]
    fh.close()
    return plevel_options


def read_variable(fname, vname, stat=None, level=None, plevel=None, region=0):
    """
    Used to read in a variable from obs_seq file

    TO-DO:
    Could extend this code with other potential dimensions in list(ds.dims)
    e.g. hlevel, but we don't need to for now

    if/else blocks should probably be try/except
    
    Inputs:
    fname (str) --> name of obs_diag_output.nc file
    vname (str) --> name of variable to be read
    stat (str, optional) --> what statistic or "copy" to be read for chosen variable
    level (int, optional) --> what pressure level to select by index (n.b. don't use both level and plevel inputs)
    plevel (int or str, optional) --> what pressure level to select by value
    region(int) --> index of region to select. defaults to 0 as we just have 1 region. If no region use None.

    Outputs:
    var (xarray DataArray)
    """
    ds = xr.open_dataset(fname)
    var = ds[vname]

    if stat:
        # Find index of selected "copy" of the variable (i.e. which statistic is wanted)
        # Probably can do this in xarray but was simpler using netCDF4
        cmd_list = list_stat_options(fname)
        stat_idx = cmd_list.index(stat)
        if "copy" in list(ds.dims):
            var = var.isel(copy=stat_idx)
        else:
            print("oops, no 'copy' options for this variable")

    if level != None:
        if "plevel" in list(ds.dims):
            var = var.isel(plevel=level)
        else:
            print("oops, no 'plevel' options for this variable")
            
    if plevel != None:
        if "plevel" in list(ds.dims):
            var = var.sel(plevel=plevel)
        else:
            print("oops, no 'plevel' options for this variable")

    if region is not None:
        if "region" in list(ds.dims):
            var = var.isel(region=0)
        else:
             print("oops, no 'region' options for this variable")

    return(var)


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


def plot_evolution(fname, vname, stat, level=None, plevel=None, fig=None, region=0, style='-', legend=False, color='turbo_r'):
    """
    Used to plot observation space time evolution from DART filter diagnostics output
    
    Inputs:
    fname (str) --> name of obs_diag_output.nc file
    vname (str) --> name of variable to be read
    stat (str, optional) --> what statistic or "copy" to be read for chosen variable
    level (int or list of ints) pressure level to select by index(s)
    plevel (int, str or list of ints/strs) pressure level to select by value(s)
    region(int) --> index of region to select. defaults to 0 as we just have 1 region. If no region use None.
    style(str) --> line style e.g. '-'
    legend (boolean)--> if True plots a legend
    color --> can specify line color as described in obs.get_rgba_color
    
    Outputs:
    fig, ax1 --> plot figure and axis
    """
    if fig:
        overlay = True
        ax1 = fig.axes[0]
        ax2 = ax1.twinx()
    else:
        overlay = False
        fig, ax1 = plt.subplots(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        
    if level != None:
        if not isinstance(level, list):
            level = [level]
        colors = get_rgba_color(color, n=len(level))
        lines=[]
        for i, l in enumerate(level):
            var = read_variable(fname, vname, stat, level=l)
            line = plt.plot(var['time'], var, style, marker='.', markersize=8, color=colors[i])[0]
            lines.append(line)
    elif plevel != None:
        colors = get_rgba_color(color, n=len(plevel))
        if not isinstance(plevel, list):
            plevel = [plevel]
        lines=[]
        for i, p in enumerate(plevel):
            var = read_variable(fname, vname, stat, plevel=p)
            line = ax1.plot(var['time'], var, style, marker='.', markersize=8, color=colors[i])[0]
            lines.append(line)

    #https://matplotlib.org/stable/gallery/text_labels_and_annotations/date.html
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    if level != None:
        plevel_options = list_plevel_options(fname)
        plt.title(f'{vname}\n@{list(plevel_options[i] for i in level)}', fontsize=16)
    elif plevel != None:
        plt.title(f'{vname}\n@{plevel} hPa', fontsize=16)
    
    ax1.set_xlabel('Date', fontsize=16)
    ax1.set_ylabel(stat, fontsize=16)
    
    for label in ax1.xaxis.get_majorticklabels():
        label.set_size(14)
    for label in ax1.yaxis.get_majorticklabels():
        label.set_size(14)
    #for label in ax1.get_xticklabels(which='major'):
    #    label.set(rotation=30, horizontalalignment='right')

    if not overlay:
        if legend:
            if level != None:
                ax1.legend(handles=lines, labels=list(plevel_options[i] for i in level), fontsize=16, reverse=True, bbox_to_anchor=(1, 0.5), loc='center left')
            elif plevel != None:
                ax1.legend(handles=lines, labels=plevel, fontsize=16, reverse=True, bbox_to_anchor=(1, 0.5), loc='center left')
    if overlay:
        ax2.tick_params(axis='y', which='both', right=False, labelright=False)
        ax1.set_ylim(min([ax1.get_ylim()[0], ax2.get_ylim()[0]]), max([ax1.get_ylim()[1], ax2.get_ylim()[1]]))
        ax2.set_ylim(min([ax1.get_ylim()[0], ax2.get_ylim()[0]]), max([ax1.get_ylim()[1], ax2.get_ylim()[1]]))
        
        if legend:
            if level != None:
                legend_obj = ax1.get_legend()
                ax1.legend(handles=legend_obj.legend_handles + lines,
                           labels=[x.get_text() for x in legend_obj.texts] + list(plevel_options[i] for i in level),
                           fontsize=16, reverse=True, bbox_to_anchor=(1, 0.5), loc='center left')
            elif plevel != None:
                legend_obj = ax1.get_legend()
                ax1.legend(handles=legend_obj.legend_handles + lines,
                           labels=[x.get_text() for x in legend_obj.texts] + plevel,
                           fontsize=16, reverse=True, bbox_to_anchor=(1, 0.5), loc='center left')

    ax1.grid(color = [0.5, 0.5, 0.5], linestyle = '--', linewidth = 0.4)
    
    return fig, ax1


def plot_profile(fname, vname, stat, region=0, fig=None, legend=False, color='k'):
    """
    Used to plot observation space time evolution from DART filter diagnostics output
    
    Inputs:
    fname (str) --> name of obs_diag_output.nc file
    vname (str) --> name of variable to be read
    stat (str, optional) --> what statistic or "copy" to be read for chosen variable
    region(int) --> index of region to select. defaults to 0 as we just have 1 region. If no region use None.
    fig --> optional figure handle to plot over the top of (e.g. from previous plot_profile call)
    legend (boolean)--> if True plots a legend
    color --> can specify line color as described in obs.get_rgba_color
    
    Outputs:
    fig, ax2, h --> plot figure, axis and handle of line/scatter plotted
    """
    if fig:
        overlay = True
        ax1 = fig.axes[0]
        ax2 = ax1.twiny()
    else:
        overlay = False
        fig, ax2 = plt.subplots(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        ax2.invert_yaxis()
        
    var = read_variable(fname, vname, stat)

    colors = get_rgba_color(color)

    if stat == 'Nposs':
        line = ax2.plot(var, var['plevel'], 'o', fillstyle='none', markersize=10, color=colors[0])[0]
    elif stat == 'Nused':
        ax2.plot(var, var['plevel'], '+', fillstyle='none', markersize=10, color=colors[0])[0]
        line = ax2.plot(var, var['plevel'], 'x', fillstyle='none', markersize=10, color=colors[0])[0]
    elif stat in  ['N_DARTqc_6', 'N_DARTqc_7']:
        line = ax2.plot(var, var['plevel'], '^', fillstyle='none', markersize=10, color=colors[0])[0]
    else:
        line = ax2.plot(var, var['plevel'], '-', marker='.', markersize=8, color=colors[0])[0]
    
    pedges = read_variable(fname, 'plevel_edges', region=None)
    [ax2.axhspan(pedges[x], pedges[x+1], color=[0, 0, 0], alpha=0.15, lw=0) for x in np.arange(0, len(pedges)-1, 2)]
    
    ax2.set_yticks(list(var['plevel'].values))

    for label in ax2.xaxis.get_majorticklabels():
        label.set_size(14)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_size(14)
    
    if not overlay:
        plt.title(vname.split('_VP')[0], fontsize=16)

        ax2.set_xlabel(stat, fontsize=16)
        ax2.set_ylabel('hPa', fontsize=16)

        if legend:
            ax2.legend(handles=[line], labels=[stat], fontsize=16, reverse=True, bbox_to_anchor=(1, 0.5), loc='center left')

        ax2.grid(color = [0.5, 0.5, 0.5], linestyle = '--', linewidth = 0.4)

    if overlay:
        if stat not in ['Nposs', 'Nused', 'N_DARTqc_6', 'N_DARTqc_7']:
            ax2.tick_params(axis='x', which='both', top=False, labeltop=False)
            ax1.set_xlim(min([ax1.get_xlim()[0], ax2.get_xlim()[0]]), max([ax1.get_xlim()[1], ax2.get_xlim()[1]]))
            ax2.set_xlim(min([ax1.get_xlim()[0], ax2.get_xlim()[0]]), max([ax1.get_xlim()[1], ax2.get_xlim()[1]]))
        if stat == 'Nposs':
            print('axis will only work properly if Nused plotted immediately before Nposs')
            ax2.set_xlim(min([fig.axes[-2].get_xlim()[0], ax2.get_xlim()[0]]), max([fig.axes[-2].get_xlim()[1], ax2.get_xlim()[1]]))            
            fig.axes[-2].set_xlim(min([fig.axes[-2].get_xlim()[0], ax2.get_xlim()[0]]), max([fig.axes[-2].get_xlim()[1], ax2.get_xlim()[1]]))            
        elif stat == 'N_DARTqc_6':
            print('axis will only work properly if Nused and Nposs plotted immediately before N_DARTqc_6')
            ax2.set_xlim(min([fig.axes[-2].get_xlim()[0], ax2.get_xlim()[0]]), max([fig.axes[-2].get_xlim()[1], ax2.get_xlim()[1]]))            
            fig.axes[-2].set_xlim(min([fig.axes[-2].get_xlim()[0], ax2.get_xlim()[0]]), max([fig.axes[-2].get_xlim()[1], ax2.get_xlim()[1]]))
            fig.axes[-3].set_xlim(min([fig.axes[-2].get_xlim()[0], ax2.get_xlim()[0]]), max([fig.axes[-2].get_xlim()[1], ax2.get_xlim()[1]]))
        elif stat == 'N_DARTqc_7':
            print('axis will only work properly if Nused, Nposs and N_DARTqc_6 plotted immediately before N_DARTqc_7')
            ax2.set_xlim(min([fig.axes[-2].get_xlim()[0], ax2.get_xlim()[0]]), max([fig.axes[-2].get_xlim()[1], ax2.get_xlim()[1]]))            
            fig.axes[-2].set_xlim(min([fig.axes[-2].get_xlim()[0], ax2.get_xlim()[0]]), max([fig.axes[-2].get_xlim()[1], ax2.get_xlim()[1]]))
            fig.axes[-3].set_xlim(min([fig.axes[-2].get_xlim()[0], ax2.get_xlim()[0]]), max([fig.axes[-2].get_xlim()[1], ax2.get_xlim()[1]]))
            fig.axes[-4].set_xlim(min([fig.axes[-2].get_xlim()[0], ax2.get_xlim()[0]]), max([fig.axes[-2].get_xlim()[1], ax2.get_xlim()[1]]))
            
        if legend:
            legend_obj = ax1.get_legend()
            ax1.legend(handles=legend_obj.legend_handles + [line],
                       labels=[x.get_text() for x in legend_obj.texts] + [stat],
                       fontsize=16, reverse=True, bbox_to_anchor=(1, 0.5), loc='center left')

        for label in ax2.yaxis.get_majorticklabels():
            label.set_size(14)

    # Saving handle for ability to make custom legends
    h = line
    
    return fig, ax2, h


def plot_n(fname, vname, level=None, plevel=None, region=0, fig=None, legend=False, color='turbo_r', n=False):
    """

    Used to plot time evolution of number of observations available and used from DART filter diagnostics output
    
    Inputs:
    fname (str) --> name of obs_diag_output.nc file
    vname (str) --> name of variable to be read
    level (int or list of ints) pressure level to select by index(s)
    plevel (int, str or list of ints/strs) pressure level to select by value(s)
    region(int) --> index of region to select. defaults to 0 as we just have 1 region. If no region use None.
    fig --> figure handle to plot second axis on top of of e.g. one from plot_evolution
    legend (boolean)--> if True plots a legend
    color --> can specify line color as described in obs.get_rgba_color
    n (integer or list) --> can be combined with color as described in obs.get_rgba_color
    
    Outputs:
    fig, ax2 --> plot figure and axis
    """
    
    if fig:
        overlay = True
        ax1 = fig.axes[0]
        ax2 = ax1.twinx()
    else:
        overlay = False
        fig, ax2 = plt.subplots(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    
    if level != None:
        if color in list(colormaps):
            n = len(level)
        colors = get_rgba_color(color, n)
        if not isinstance(level, list):
            level = [level]
        scatters=[]
        for i, l in enumerate(level):
            var = read_variable(fname, vname, 'Nposs', level=l)
            scatter = ax2.plot(var['time'], var, 'o', fillstyle='none', markersize=10, color=colors[i])[0]
            scatters.append(scatter)
            var = read_variable(fname, vname, 'Nused', level=l)
            ax2.plot(var['time'], var, '+', fillstyle='none', markersize=10, color=colors[i])
            ax2.plot(var['time'], var, 'x', fillstyle='none', markersize=10, color=colors[i])
    elif plevel != None:
        if color in list(colormaps):
            n = len(plevel)
        colors = get_rgba_color(color, n)
        if not isinstance(plevel, list):
            plevel = [plevel]
        scatters=[]
        for i, p in enumerate(plevel):
            var = read_variable(fname, vname, 'Nposs', plevel=p)
            scatter = ax2.plot(var['time'], var, 'o', fillstyle='none', markersize=10, color=colors[i])[0]
            scatters.append(scatter)
            var = read_variable(fname, vname, 'Nused', plevel=p)
            ax2.plot(var['time'], var, '+', fillstyle='none', markersize=10, color=colors[i])
            ax2.plot(var['time'], var, 'x', fillstyle='none', markersize=10, color=colors[i])

    if not overlay:
        #https://matplotlib.org/stable/gallery/text_labels_and_annotations/date.html
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        if level != None:
            plevel_options = list_plevel_options(fname)
            plt.title(f'{vname}\n@{list(plevel_options[i] for i in level)}', fontsize=16)
        elif plevel != None:
            plt.title(f'{vname}\n@{plevel} hPa', fontsize=16)

        ax2.set_xlabel('Date', fontsize=16)
        
        for label in ax2.xaxis.get_majorticklabels():
            label.set_size(14)
        #for label in ax2.get_xticklabels(which='major'):
        #    label.set(rotation=30, horizontalalignment='right')

        if legend:
            if level != None:
                ax2.legend(handles=scatters, labels=list(plevel_options[i] for i in level), fontsize=16, reverse=True, bbox_to_anchor=(1, 0.5), loc='center left')
            elif plevel:
                ax2.legend(handles=scatters, labels=plevel, fontsize=16, reverse=True, bbox_to_anchor=(1, 0.5), loc='center left')

        ax2.grid(color = [0.5, 0.5, 0.5], linestyle = '--', linewidth = 0.4)

    if overlay:
        if legend:
            legend_obj = ax1.get_legend()
            legend_obj.set_bbox_to_anchor((1.1, 0.5))
    
    ax2.set_ylabel('# observations: o=possible, *=assimilated', fontsize=16)

    for label in ax2.yaxis.get_majorticklabels():
        label.set_size(14)
    
    return fig, ax2


def plot_rank_hist(fname, vname, level=None, plevel=None, region=0, time=None, scale=False,
                    style="bar", legend=False, color='turbo_r'):
    """
    Used to plot rank histograms from DART filter diagnostics output
    
    Inputs:
    fname (str) --> name of obs_diag_output.nc file
    vname (str) --> name of variable to be read
    level (int or list of ints) pressure level to select by index(s)
    plevel (int, str or list of ints/strs) pressure level to select by value(s)
    region (int) --> index of region to select. defaults to 0 as we just have 1 region.
    legend (boolean)--> if True plots a legend
    time --> either (int) for selecting index, (np.datetime64) for selecting time
         --> can also be (list of ints) or (list of np.datetime64) to plot multiple times
             --> see create_time_list for list of times creation function
    scale (boolean) --> option to scale counts by total count across all ranks
                    --> a consistent value of 1 on the count axis is desirable after this scaling
    style (string) --> specifies histogram style from "bar", "step", "hlines", "line", "linedot",
                   --> "box" to plot box plots
                   --> or you can try a marker syle such as 'o'. some styles may require code edits.
    legend (boolean) --> currently unused
    color --> can specify line color as described in obs.get_rgba_color
    
    Outputs:
    fig, ax1 --> plot figure and axis
    """

    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

    if level != None:
        level_option = level
    elif plevel != None:
        level_option = plevel
    
    if isinstance(level_option, (int, float, str)):
        level_option = [level_option]

    if isinstance(time, (int, float, str)):
        time = [time]

    if color in list(colormaps):
        colors = get_rgba_color(color, len(time) * len(level_option))
    else:
        colors = get_rgba_color(color)
        colors = np.tile(colors, (len(time) * len(level_option), 1))

    if not isinstance(time, np.ndarray):
        if time == None:
            time = [float(0)] #to avoid errors when indexing into time
    
    if style == "box":
        if level != None:
            counts = read_variable(fname, vname, level=level_option)
        elif plevel != None:
            counts = read_variable(fname, vname, plevel=level_option)

        if isinstance(time[0], int):
                counts = counts[t, :]
        elif isinstance(time[0], np.datetime64):
            counts = counts.sel(time=time)

        counts = counts.squeeze()
        if scale:
            counts = counts * len(counts['rank_bins']) / counts.sum(dim="rank_bins")
            ax1.hlines(1, 0, len(counts['rank_bins']) + 1, color='k', linestyle='--')
        filtered_data = [d[m] for d, m in zip(counts.T, ~np.isnan(counts).T)]
        
        ax1.boxplot(filtered_data)
        
    else:
        cidx = -1
        for t, lev in itertools.product(time, level_option):
            cidx+=1
            
            if level != None:
                counts = read_variable(fname, vname, level=lev)
            elif plevel != None:
                counts = read_variable(fname, vname, plevel=lev)
    
            if isinstance(t, int):
                counts = counts[t, :]
            elif isinstance(t, np.datetime64):
                counts = counts.sel(time=t)
    
            if scale:
                counts = counts * len(counts['rank_bins']) / np.nansum(counts)  
            
            if style == "bar":
                ax1.bar(range(1, len(counts) + 1), counts, width=1.0, color=colors[cidx], edgecolor='black')
            elif style == "step":
                ax1.step(range(1, len(counts) + 1), counts, color=colors[cidx], where="mid")
            elif style == "hlines":
                ax1.hlines(counts,
                           [x - 0.5 for x in list(range(1, len(counts) + 1))],
                           [x + 0.5 for x in list(range(1, len(counts) + 1))],
                           color=colors[cidx]
                          )
            elif style == "line":
                ax1.plot(range(1, len(counts) + 1), counts, '-',  color=colors[cidx])
            elif style == "linedot":
                ax1.plot(range(1, len(counts) + 1), counts, '-',  color=colors[cidx])
                ax1.plot(range(1, len(counts) + 1), counts, 'o',  color=colors[cidx])
            else:
                try:
                    ax1.plot(range(1, len(counts) + 1), counts, style,
                             markeredgecolor=colors[cidx], markerfacecolor ='none')
                except:
                    print("Error: can't find valid plot or marker style. Exiting.")
                    return
    
    for label in ax1.xaxis.get_majorticklabels():
        label.set_size(14)
    for label in ax1.yaxis.get_majorticklabels():
        label.set_size(14)
    
    plt.title(f"{vname.split('_guess')[0]}\n@{plevel} hPa", fontsize=16)
    
    ax1.set_xlabel('Observation rank (among ensemble members)', fontsize=16)
    ax1.set_ylabel('Count', fontsize=16)
    
    ax1.set_xlim([0, len(counts['rank_bins']) + 1])
    ax1.set_ylim([0, ax1.get_ylim()[1]])
    
    ax1.yaxis.grid(True)

    return fig, ax1


def create_time_list(start_time, end_time, time_step_hrs):
    """
    Creates a list of np.datetime64 times which can be fed into rank histogram plots
    Start and end times are incuded in range returned

    Inputs:
    start_time --> string with format YYYY-MM-DDTHH e.g. '2020-08-08T06'
    end_time --> string with format YYYY-MM-DDTHH e.g. '2020-08-08T06'
    time_step_hrs --> integer number of hours between elements in list e.g. 6

    Returns:
    time_list --> list of times in np.datetime64 format
    """
    
    time_list = np.arange(np.datetime64(start_time),
                          np.datetime64(end_time) + np.timedelta64(1, 'ns'),
                          np.timedelta64(time_step_hrs, 'h'))
    return time_list