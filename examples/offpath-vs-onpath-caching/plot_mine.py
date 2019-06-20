
#!/usr/bin/env python
"""Plot results read from a result set
"""
from __future__ import division
import os
import collections

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import argparse
import logging


from icarus.util import Settings, Tree, config_logging, step_cdf
from icarus.tools import means_confidence_interval
from icarus.registry import RESULTS_READER



#__all__ = ['plot_lines']


# These lines prevent insertion of Type 3 fonts in figures
# Publishers don't want them. However, in some case these commands block the
# embedding of fonts raising complaints for example from EDAS
# plt.rcParams['ps.useafm'] = True
# plt.rcParams['pdf.use14corefonts'] = True

# If True text is interpreted as LaTeX, e.g. underscore are interpreted as
# subscript. If False, text is interpreted literally
plt.rcParams['text.usetex'] = False

# Aspect ratio of the output figures
plt.rcParams['figure.figsize'] = 8, 5

# Size of font in legends
LEGEND_SIZE = 14

# Plot
PLOT_EMPTY_GRAPHS = False

# Catalogue of possible bw shades (for bar charts)
BW_COLOR_CATALOGUE = ['k', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

# Catalogue of possible hatch styles (for bar charts)
HATCH_CATALOGUE = [None, '/', '\\', '\\\\', '//', '+', 'x', '*', 'o', '.', '|', '-', 'O']
'''

def plot_lines(resultset, desc, filename, plotdir , subnum):
    """Plot a graph with characteristics described in the plot descriptor out
    of the data contained in the resultset and save the plot in given directory.

    Parameters
    ----------
    rs : ResultSet
        Result set
    desc : dict
        The plot descriptor (more info below)
    filename : str
        The name used to save the file. The file format is determined by the
        extension of the file. For example, if this filename is 'foo.pdf', the
        file will be saved in pdf format.
    plotdir : str
        The directory in which the plot will be saved.

    Notes
    -----
    The plot descriptor is a dictionary with a set of values that describe how
    to make the plot.

    The dictionary can contain the following keys:
     * title : str, optional.
           The title of the graph
     * xlabel : str, optional
         The x label
     * ylabel : str, optional
         The y label
     * errorbar : bool, optional
         If *True* error bars will be plotted. Default value is *True*
     * confidence : float, optional
         The confidence used to plot error bars. Default value is 0.95
     * xparam : iterable
         Path to the value of the x axis metric, e.g. ['workload', 'alpha']
     * xvals : list
         Range of x values, e.g. [0.6, 0.7, 0.8, 0.9]
     * filter : dict, optional
         A dictionary of values to filter in the resultset.
         Example: {'network_cache': 0.004, 'topology_name': 'GEANT'}
         If not specified or None, no filtering is executed on the results
         and possibly heterogeneous results may be plotted together
     * ymetrics : list of tuples
         List of metrics to be shown on the graph. The i-th metric of the list
         is the metric that the i-th line on the graph will represent. If
         all lines are for the same metric, then all elements of the list are
         equal.
         Each single metric (i.e. each element of the list) is a tuple modeling
         the path to identify a specific metric into an entry of a result set.
         Normally, it is a 2-value list where the first value is the name of
         the collector which measured the metric and the second value is the
         metric name. Example values could be ('CACHE_HIT_RATIO', 'MEAN'),
         ('LINK_LOAD', 'MEAN_INTERNAL') or ('LATENCY', 'MEAN').
         For example, if in a graph of N lines all lines of the graph show mean
         latency, then ymetrics = [('LATENCY', 'MEAN')]*5.
     * ycondnames : list of tuples, optional
         List of condition names specific to each line of the graph. Different
         from the conditions expressed in the filter parameter, which are
         global, these conditions are specific to one bar. Ech condition name,
         different from the filter parameter is a path to a condition to be
         checked, e.g. ('topology', 'name'). Values to be matched for this
         conditions are specified in ycondvals. This list must be as long as
         the number of lines to plot. If not specified, all lines are filtered
         by the conditions of filter parameter only, but in this case all
         ymetrics should be different.
     * ycondvals : list of tuples, optional
         List of values that the conditions of ycondnames must meet. This list
         must be as long as the number of lines to plot. If not specified,
         all lines are filtered by the conditions of filter parameter only,
         but in this case all ymetrics should be different.
     * xscale : ('linear' | 'log'), optional
         The scale of x axis. Default value is 'linear'
     * yscale : ('linear' | 'log'), optional
         The scale of y axis. Default value is 'linear'
     * xticks : list, optional
         Values to display as x-axis ticks.
     * yticks : list, optional
         Values to display as y-axis ticks.
     * line_style : dict, optional
         Dictionary mapping each value of yvals with a line style
     * plot_args : dict, optional
         Additional args to be provided to the Pyplot errorbar function.
         Example parameters that can be specified here are *linewidth* and
         *elinewidth*
     * legend : dict, optional
         Dictionary mapping each value of yvals with a legend label. If not
         specified, it is not plotted. If you wish to plot it with the
         name of the line, set it to put yvals or ymetrics, depending on which
         one is used
     * legend_loc : str, optional
         Legend location, e.g. 'upper left'
     * legend_args : dict, optional
         Optional legend arguments, such as ncol
     * plotempty : bool, optional
         If *True*, plot and save graph even if empty. Default is *True*
     * xmin, xmax: float, optional
        The limits of the x axis. If not specified, they're set to the min and
        max values of xvals
     * ymin, ymax: float, optional
        The limits of the y axis. If not specified, they're automatically
        selected by Matplotlib
    """
    subid = '14' + str(subnum)  

    fig = plt.figure()
    ax = fig.add_subplot(subid)
    if 'title' in desc:
        plt.title(desc['title'])
    if 'xlabel' in desc:
        plt.xlabel(desc['xlabel'])
    if 'ylabel' in desc:
        plt.ylabel(desc['ylabel'])
    if 'xscale' in desc:
        plt.xscale(desc['xscale'])
    if 'yscale' in desc:
        plt.yscale(desc['yscale'])
    if 'filter' not in desc or desc['filter'] is None:
        desc['filter'] = {}
    xvals = sorted(desc['xvals'])
    if 'xticks' in desc:
        ax1.set_xticks(desc['xticks'])
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.set_xticklabels([str(xtick) for xtick in desc['xticks']])
    if 'yticks' in desc:
        ax1.set_yticks(desc['yticks'])
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.set_yticklabels([str(ytick) for ytick in desc['yticks']])
    ymetrics = desc['ymetrics']
    ycondnames = desc['ycondnames'] if 'ycondnames' in desc else None
    ycondvals = desc['ycondvals'] if 'ycondvals' in desc else None
    if ycondnames is not None and ycondvals is not None:
        if not len(ymetrics) == len(ycondnames) == len(ycondvals):
            raise ValueError('ymetrics, ycondnames and ycondvals must have the same length')
        # yvals is basically the list of values that differentiate each line
        # it is used for legends and styles mainly
        yvals = ycondvals if len(set(ymetrics)) == 1 else zip(ymetrics, ycondvals)
    else:
        yvals = ymetrics
    plot_args = desc['plot_args'] if 'plot_args' in desc else {}
    plot_empty = desc['plotempty'] if 'plotempty' in desc else True
    empty = True
    for i in range(len(yvals)):
        means = np.zeros(len(xvals))
        err = np.zeros(len(xvals))
        for j in range(len(xvals)):
            condition = Tree(desc['filter'])
            condition.setval(desc['xparam'], xvals[j])
            if ycondnames is not None:
                condition.setval(ycondnames[i], ycondvals[i])
            data = [v.getval(ymetrics[i])
                    for _, v in resultset.filter(condition)
                    if v.getval(ymetrics[i]) is not None]
            confidence = desc['confidence'] if 'confidence' in desc else 0.95
            means[j], err[j] = means_confidence_interval(data, confidence)
        yerr = None if 'errorbar' in desc and not desc['errorbar'] or all(err == 0) else err
        fmt = desc['line_style'][yvals[i]] if 'line_style' in desc \
              and yvals[i] in desc['line_style'] else '-'
        # This check is to prevent crashing when trying to plot arrays of nan
        # values with axes log scale
        if all(np.isnan(x) for x in xvals) or all(np.isnan(y) for y in means):
            plt.errorbar([], [], fmt=fmt)
        else:
            plt.errorbar(xvals, means, yerr=yerr, fmt=fmt, **plot_args)
            empty = False
    if empty and not plot_empty:
        return
    x_min = desc['xmin'] if 'xmin' in desc else min(xvals)
    x_max = desc['xmax'] if 'xmax' in desc else max(xvals)
    plt.xlim(x_min, x_max)
    if 'ymin' in desc:
        plt.ylim(ymin=desc['ymin'])
    if 'ymax' in desc:
        plt.ylim(ymax=desc['ymax'])
    if 'legend' in desc:
        legend = [desc['legend'][l] for l in yvals]
        legend_args = desc['legend_args'] if 'legend_args' in desc else {}
        if 'legend_loc' in desc:
            legend_args['loc'] = desc['legend_loc']
        plt.legend(legend, prop={'size': LEGEND_SIZE}, **legend_args)
    plt.savefig(os.path.join(plotdir, filename), bbox_inches='tight')
    plt.close(fig)

'''



# Logger object
logger = logging.getLogger('plot')

# These lines prevent insertion of Type 3 fonts in figures
# Publishers don't want them
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True

# If True text is interpreted as LaTeX, e.g. underscore are interpreted as 
# subscript. If False, text is interpreted literally
plt.rcParams['text.usetex'] = False

# Aspect ratio of the output figures
plt.rcParams['figure.figsize'] = 8, 5

# Size of font in legends
LEGEND_SIZE = 14

# Line width in pixels
LINE_WIDTH = 1.5

# Plot
PLOT_EMPTY_GRAPHS = True

# This dict maps strategy names to the style of the line to be used in the plots
# Off-path strategies: solid lines
# On-path strategies: dashed lines
# No-cache: dotted line
STRATEGY_STYLE = {
         'HR_SYMM':         'b-o',
         'HR_ASYMM':        'g-D',
         'HR_MULTICAST':    'm-^',         
         'HR_HYBRID_AM':    'c-s',
         'HR_HYBRID_SM':    'r-v',
         'LCE':             'b--p',
         'LCD':             'g-->',
         'CL4M':            'g-->',
         'PROB_CACHE':      'c--<',
         'RAND_CHOICE':     'r--<',
         'RAND_BERNOULLI':  'g--*',
         'NO_CACHE':        'k:o',
         'OPTIMAL':         'k-o'
                }

# This dict maps name of strategies to names to be displayed in the legend
STRATEGY_LEGEND = {
         'LCE':             'LCE',
         'LCD':             'LCD',
         'HR_SYMM':         'HR Symm',
         'HR_ASYMM':        'HR Asymm',
         'HR_MULTICAST':    'HR Multicast',         
         'HR_HYBRID_AM':    'HR Hybrid AM',
         'HR_HYBRID_SM':    'HR Hybrid SM',
         'CL4M':            'CL4M',
         'PROB_CACHE':      'ProbCache',
         'RAND_CHOICE':     'Random (choice)',
         'RAND_BERNOULLI':  'Random (Bernoulli)',
         'NO_CACHE':        'No caching',
         'OPTIMAL':         'Optimal'
                    }

# Color and hatch styles for bar charts of cache hit ratio and link load vs topology
STRATEGY_BAR_COLOR = {
    'LCE':          'k',
    'LCD':          '0.4',
    'NO_CACHE':     '0.5',
    'HR_ASYMM':     '0.6',
    'HR_SYMM':      '0.7'
    }

STRATEGY_BAR_HATCH = {
    'LCE':          None,
    'LCD':          '//',
    'NO_CACHE':     'x',
    'HR_ASYMM':     '+',
    'HR_SYMM':      '\\'
    }




def plot_cache_hits_vs_alpha(resultset, topology, cache_size, alpha_range, strategies, plotdir,i):
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc = {}
    desc['title'] = 'Cache hit ratio: T=%s C=%s' % (topology, cache_size)
    desc['ylabel'] = 'Cache hit ratio'
    desc['xlabel'] = u'Content distribution \u03b1'
    desc['xparam'] = ('workload', 'alpha')
    desc['xvals'] = alpha_range
    desc['filter'] = {'topology': {'name': topology},
                      'cache_placement': {'network_cache': cache_size}}
    desc['ymetrics'] = [('CACHE_HIT_RATIO', 'MEAN')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper left'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    
    k = len(alpha_range)
    #filename = 'CACHE_HIT_RATIO_T=%s@C=%s.pdf'% (topology, cache_size)



    #subid = '14' + str(i+1)

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 4 , i+1)
    if 'title' in desc:
        plt.title(desc['title'])
    if 'xlabel' in desc:
        plt.xlabel(desc['xlabel'])
    if 'ylabel' in desc:
        plt.ylabel(desc['ylabel'])
    if 'xscale' in desc:
        plt.xscale(desc['xscale'])
    if 'yscale' in desc:
        plt.yscale(desc['yscale'])
    if 'filter' not in desc or desc['filter'] is None:
        desc['filter'] = {}
    xvals = sorted(desc['xvals'])
    if 'xticks' in desc:
        ax1.set_xticks(desc['xticks'])
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.set_xticklabels([str(xtick) for xtick in desc['xticks']])
    if 'yticks' in desc:
        ax1.set_yticks(desc['yticks'])
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.set_yticklabels([str(ytick) for ytick in desc['yticks']])
    ymetrics = desc['ymetrics']
    ycondnames = desc['ycondnames'] if 'ycondnames' in desc else None
    ycondvals = desc['ycondvals'] if 'ycondvals' in desc else None
    if ycondnames is not None and ycondvals is not None:
        if not len(ymetrics) == len(ycondnames) == len(ycondvals):
            raise ValueError('ymetrics, ycondnames and ycondvals must have the same length')
        # yvals is basically the list of values that differentiate each line
        # it is used for legends and styles mainly
        yvals = ycondvals if len(set(ymetrics)) == 1 else zip(ymetrics, ycondvals)
    else:
        yvals = ymetrics
    plot_args = desc['plot_args'] if 'plot_args' in desc else {}
    plot_empty = desc['plotempty'] if 'plotempty' in desc else True
    empty = True
    for i in range(len(yvals)):
        means = np.zeros(len(xvals))
        err = np.zeros(len(xvals))
        for j in range(len(xvals)):
            condition = Tree(desc['filter'])
            condition.setval(desc['xparam'], xvals[j])
            if ycondnames is not None:
                condition.setval(ycondnames[i], ycondvals[i])
            data = [v.getval(ymetrics[i])
                    for _, v in resultset.filter(condition)
                    if v.getval(ymetrics[i]) is not None]

            confidence = desc['confidence'] if 'confidence' in desc else 0.95
            means[j], err[j] = means_confidence_interval(data, confidence)
        yerr = None if 'errorbar' in desc and not desc['errorbar'] or all(err == 0) else err
        fmt = desc['line_style'][yvals[i]] if 'line_style' in desc \
              and yvals[i] in desc['line_style'] else '-'
        # This check is to prevent crashing when trying to plot arrays of nan
        # values with axes log scale
        if all(np.isnan(x) for x in xvals) or all(np.isnan(y) for y in means):
            plt.errorbar([], [], fmt=fmt)
        else:
            plt.errorbar(xvals, means, yerr=yerr, fmt=fmt, **plot_args)
            empty = False
    if empty and not plot_empty:
        return
    x_min = desc['xmin'] if 'xmin' in desc else min(xvals)
    x_max = desc['xmax'] if 'xmax' in desc else max(xvals)
    plt.xlim(x_min, x_max)
    if 'ymin' in desc:
        plt.ylim(ymin=desc['ymin'])
    if 'ymax' in desc:
        plt.ylim(ymax=desc['ymax'])
    if 'legend' in desc:
        legend = [desc['legend'][l] for l in yvals]
        legend_args = desc['legend_args'] if 'legend_args' in desc else {}
        if 'legend_loc' in desc:
            legend_args['loc'] = desc['legend_loc']
        plt.legend(legend, prop={'size': LEGEND_SIZE}, **legend_args)
    #plt.savefig(os.path.join(plotdir, filename), bbox_inches='tight')
    #plt.close(fig)

'''
def plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_size_range, strategies, plotdir):
    desc = {}
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc['title'] = 'Cache hit ratio: T=%s A=%s' % (topology, alpha)
    desc['xlabel'] = u'Cache to population ratio'
    desc['ylabel'] = 'Cache hit ratio'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement','network_cache')
    desc['xvals'] = cache_size_range
    desc['filter'] = {'topology': {'name': topology},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha}}
    desc['ymetrics'] = [('CACHE_HIT_RATIO', 'MEAN')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper left'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    k = len(cache_size_range)
    plot_lines(resultset, desc,'CACHE_HIT_RATIO_T=%s@A=%s.pdf'
               % (topology, alpha), plotdir,k)
    

def plot_link_load_vs_alpha(resultset, topology, cache_size, alpha_range, strategies, plotdir):
    desc = {}
    desc['title'] = 'Internal link load: T=%s C=%s' % (topology, cache_size)
    desc['xlabel'] = u'Content distribution \u03b1'
    desc['ylabel'] = 'Internal link load'
    desc['xparam'] = ('workload', 'alpha')
    desc['xvals'] = alpha_range
    desc['filter'] = {'topology': {'name': topology},
                      'cache_placement': {'network_cache': cache_size}}
    desc['ymetrics'] = [('LINK_LOAD', 'MEAN_INTERNAL')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    k = len(alpha_range)
    plot_lines(resultset, desc, 'LINK_LOAD_INTERNAL_T=%s@C=%s.pdf'
               % (topology, cache_size), plotdir,k)


def plot_link_load_vs_cache_size(resultset, topology, alpha, cache_size_range, strategies, plotdir):
    desc = {}
    desc['title'] = 'Internal link load: T=%s A=%s' % (topology, alpha)
    desc['xlabel'] = 'Cache to population ratio'
    desc['ylabel'] = 'Internal link load'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement','network_cache')
    desc['xvals'] = cache_size_range
    desc['filter'] = {'topology': {'name': topology},
                      'workload': {'name': 'stationary', 'alpha': alpha}}
    desc['ymetrics'] = [('LINK_LOAD', 'MEAN_INTERNAL')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    k = len(cache_size)
    plot_lines(resultset, desc, 'LINK_LOAD_INTERNAL_T=%s@A=%s.pdf'
               % (topology, alpha), plotdir,k)
    

def plot_latency_vs_alpha(resultset, topology, cache_size, alpha_range, strategies, plotdir):
    desc = {}
    desc['title'] = 'Latency: T=%s C=%s' % (topology, cache_size)
    desc['xlabel'] = u'Content distribution \u03b1'
    desc['ylabel'] = 'Latency (ms)'
    desc['xparam'] = ('workload', 'alpha')
    desc['xvals'] = alpha_range
    desc['filter'] = {'topology': {'name': topology},
                      'cache_placement': {'network_cache': cache_size}}
    desc['ymetrics'] = [('LATENCY', 'MEAN')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    k = len(alpha_range)
    plot_lines(resultset, desc, 'LATENCY_T=%s@C=%s.pdf'
               % (topology, cache_size), plotdir,k)


def plot_latency_vs_cache_size(resultset, topology, alpha, cache_size_range, strategies, plotdir):
    desc = {}
    desc['title'] = 'Latency: T=%s A=%s' % (topology, alpha)
    desc['xlabel'] = 'Cache to population ratio'
    desc['ylabel'] = 'Latency'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement','network_cache')
    desc['xvals'] = cache_size_range
    desc['filter'] = {'topology': {'name': topology},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha}}
    desc['ymetrics'] = [('LATENCY', 'MEAN')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['metric'] = ('LATENCY', 'MEAN')
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    k = len(cache_size_range)
    plot_lines(resultset, desc, 'LATENCY_T=%s@A=%s.pdf'
               % (topology, alpha), plotdir,k)
    
'''


def run(config, results, plotdir):
    """Run the plot script
    
    Parameters
    ----------
    config : str
        The path of the configuration file
    results : str
        The file storing the experiment results
    plotdir : str
        The directory into which graphs will be saved
    """
    settings = Settings()
    settings.read_from(config)
    config_logging(settings.LOG_LEVEL)
    resultset = RESULTS_READER[settings.RESULTS_FORMAT](results)
    # Create dir if not existsing
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    # Parse params from settings
    topologies = settings.TOPOLOGIES
    cache_sizes = settings.NETWORK_CACHE
    alphas = settings.ALPHA
    strategies = settings.STRATEGIES
    # Plot graphs
    for topology in topologies:
        for i in range(1 ,5):
            fig = plt.figure()
            ax = fig.add_subplot(1, 4 , i)
            for cache_size in cache_sizes:
                logger.info('Plotting cache hit ratio for topology %s and cache size %s vs alpha' % (topology, str(cache_size)))
                plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, strategies, plotdir,i)

            filename = 'CACHE_HIT_RATIO_T=%s@C=%s.pdf'% (topology, cache_size)
            plt.savefig(os.path.join(plotdir, filename), bbox_inches='tight')
        plt.close(fig)

            
                #logger.info('Plotting link load for topology %s vs cache size %s' % (topology, str(cache_size)))
                #plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, strategies, plotdir,i)
                #logger.info('Plotting latency for topology %s vs cache size %s' % (topology, str(cache_size)))
                #plot_latency_vs_alpha(resultset, topology, cache_size, alphas, strategies, plotdir,i)
    '''
    for topology in topologies:
        for i in range(len(toplogies)):
            for alpha in alphas:
                logger.info('Plotting cache hit ratio for topology %s and alpha %s vs cache size' % (topology, str(alpha)))
                plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, strategies, plotdir,i)
                logger.info('Plotting link load for topology %s and alpha %s vs cache size' % (topology, str(alpha)))
                plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, strategies, plotdir,i)
                logger.info('Plotting latency for topology %s and alpha %s vs cache size' % (topology, str(alpha)))
                plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, strategies, plotdir,i)
    logger.info('Exit. Plots were saved in directory %s' % os.path.abspath(plotdir))
'''


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-r", "--results", dest="results",
                        help='the results file',
                        required=True)
    parser.add_argument("-o", "--output", dest="output",
                        help='the output directory where plots will be saved',
                        required=True)
    parser.add_argument("config",
                        help="the configuration file")
    args = parser.parse_args()
    run(args.config, args.results, args.output)

if __name__ == '__main__':
    main()
