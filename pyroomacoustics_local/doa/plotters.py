'''
A collection of functions to plot maps and points on circles and spheres.
'''
import numpy as np

def polar_plt_dirac(self, azimuth_ref=None, alpha_ref=None, save_fig=False, 
        file_name=None, plt_dirty_img=True):
    """
    Generate polar plot of DoA results.

    Parameters
    ----------
    azimuth_ref: numpy array
        True direction of sources (in radians).
    alpha_ref: numpy array
        Estimated amplitude of sources.
    save_fig: bool
        Whether or not to save figure as pdf.
    file_name: str
        Name of file (if saved). Default is 
        'polar_recon_dirac.pdf'
    plt_dirty_img: bool
        Whether or not to plot spatial spectrum or 
        'dirty image' in the case of FRI.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings
        warnings.warn('Matplotlib is required for plotting')
        return

    if self.dim != 2:
        raise ValueError('This function only handles 2D problems.')

    azimuth_recon = self.azimuth_recon
    num_mic = self.M
    phi_plt = self.grid.azimuth

    # determine amplitudes
    from fri import FRI
    if not isinstance(self, FRI):  # use spatial spectrum

        dirty_img = self.grid.values
        alpha_recon = self.grid.values[self.src_idx]
        alpha_ref = alpha_recon

    else:  # create dirty image

        dirty_img = self._gen_dirty_img()
        alpha_recon = np.mean(np.abs(self.alpha_recon), axis=1)
        alpha_recon /= alpha_recon.max()
        if alpha_ref is None:   # non-simulated case
            alpha_ref = alpha_recon

    # plot
    fig = plt.figure(figsize=(5, 4), dpi=90)
    ax = fig.add_subplot(111, projection='polar')
    base = 1.
    height = 10.
    blue = [0, 0.447, 0.741]
    red = [0.850, 0.325, 0.098]

    if azimuth_ref is not None:
        if alpha_ref.shape[0] < azimuth_ref.shape[0]:
            alpha_ref = np.concatenate((alpha_ref,np.zeros(azimuth_ref.shape[0]-
                alpha_ref.shape[0])))

        # match detected with truth
        recon_err, sort_idx = polar_distance(azimuth_recon, azimuth_ref)
        if self.num_src > 1:
            azimuth_recon = azimuth_recon[sort_idx[:, 0]]
            alpha_recon = alpha_recon[sort_idx[:, 0]]
            azimuth_ref = azimuth_ref[sort_idx[:, 1]]
            alpha_ref = alpha_ref[sort_idx[:, 1]]
        elif azimuth_ref.shape[0] > 1:   # one detected source
            alpha_ref[sort_idx[1]] =  alpha_recon

        # markers for original doa
        K = len(azimuth_ref)
        ax.scatter(azimuth_ref, base + height*alpha_ref, c=np.tile(blue, 
            (K, 1)), s=70, alpha=0.75, marker='^', linewidths=0, 
            label='original')

        # stem for original doa
        if K > 1:
            for k in range(K):
                ax.plot([azimuth_ref[k], azimuth_ref[k]], [base, base + 
                    height*alpha_ref[k]], linewidth=1.5, linestyle='-', 
                    color=blue, alpha=0.6)
        else:
            ax.plot([azimuth_ref, azimuth_ref], [base, base + height*alpha_ref], 
                linewidth=1.5, linestyle='-', color=blue, alpha=0.6)


    K_est = azimuth_recon.size

    # markers for reconstructed doa
    ax.scatter(azimuth_recon, base + height*alpha_recon, c=np.tile(red, 
        (K_est, 1)), s=100, alpha=0.75, marker='*', linewidths=0, 
        label='reconstruction')

    # stem for reconstructed doa
    if K_est > 1:
        for k in range(K_est):
            ax.plot([azimuth_recon[k], azimuth_recon[k]], [base, base + 
                height*alpha_recon[k]], linewidth=1.5, linestyle='-', 
                color=red, alpha=0.6)

    else:
        ax.plot([azimuth_recon, azimuth_recon], [base, base + height*alpha_recon], 
            linewidth=1.5, linestyle='-', color=red, alpha=0.6)            

    # plot the 'dirty' image
    if plt_dirty_img:
        dirty_img = np.abs(dirty_img)
        min_val = dirty_img.min()
        max_val = dirty_img.max()
        dirty_img = (dirty_img - min_val) / (max_val - min_val)

        # we need to make a complete loop, copy first value to last
        c_phi_plt = np.r_[phi_plt, phi_plt[0]]
        c_dirty_img = np.r_[dirty_img, dirty_img[0]]
        ax.plot(c_phi_plt, base + height*c_dirty_img, linewidth=1, 
            alpha=0.55,linestyle='-', color=[0.466, 0.674, 0.188], 
            label='spatial spectrum')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:3], framealpha=0.5,
              scatterpoints=1, loc=8, fontsize=9,
              ncol=1, bbox_to_anchor=(0.9, -0.17),
              handletextpad=.2, columnspacing=1.7, labelspacing=0.1)

    ax.set_xlabel(r'azimuth $\bm{\varphi}$', fontsize=11)
    ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
    ax.xaxis.set_label_coords(0.5, -0.11)
    ax.set_yticks(np.linspace(0, 1, 2))
    ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
    ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
    ax.set_ylim([0, 1.05 * (base + height)])
    if save_fig:
        if file_name is None:
            file_name = 'polar_recon_dirac.pdf'
        plt.savefig(file_name, format='pdf', dpi=300, transparent=True)


# ===========plotting functions===========
def sph_plot_diracs_plotly(
        colatitude_ref=None, azimuth_ref=None, colatitude=None, azimuth=None,
        dirty_img=None, azimuth_grid=None, colatitude_grid=None,
        surface_base=1, surface_height=0.,
        ):
    '''
    Plots a 2D map on a sphere as well as a collection of diracs using the plotly library

    Parameters
    ----------
    colatitude_ref: ndarray, optional
        The colatitudes of a collection of reference points
    azimuths_ref: ndarray, optional
        The azimuths of a collection of reference points for the Diracs
    colatitude: ndarray, optional
        The colatitudes of the collection of points to visualize
    azimuth: ndarray, optional
        The azimuths of the collection of points to visualize
    dirty_img: ndarray
        A 2D map for displaying a pattern on the sphere under the points
    azimuth_grid: ndarray
        The azimuths indexing the dirty_img 2D map
    colatitude_grid: ndarray
        The colatitudes indexing the dirty_img 2D map
    surface_base:
        radius corresponding to lowest height on the map
    sufrace_height:
        radius difference between the lowest and highest point on the map
    '''

    try:
        from plotly.offline import plot
        import plotly.graph_objs as go
        import plotly
    except ImportError:
        import warnings
        warnings.warn('The plotly package is required to use this function')
        return

    plotly.offline.init_notebook_mode()

    traces = []

    if dirty_img is not None and azimuth_grid is not None and colatitude_grid is not None:

        surfacecolor = np.abs(dirty_img)  # for plotting purposes

        base = surface_base

        surf_diff = surfacecolor.max() - surfacecolor.min()
        if surf_diff > 0:
            height = surface_height / surf_diff
        else:
            height = 0


        r_surf = base + height * (surfacecolor - surfacecolor.min()) / (surfacecolor.max() - surfacecolor.min())

        x_plt = r_surf * np.sin(colatitude_grid) * np.cos(azimuth_grid)
        y_plt = r_surf * np.sin(colatitude_grid) * np.sin(azimuth_grid)
        z_plt = r_surf * np.cos(colatitude_grid)

        trace1 = go.Surface(x=x_plt, y=y_plt, z=z_plt, surfacecolor=surfacecolor,
                            opacity=1, colorscale='Portland', hoverinfo='none')

        trace1['contours']['x']['highlightwidth'] = 1
        trace1['contours']['y']['highlightwidth'] = 1
        trace1['contours']['z']['highlightwidth'] = 1

        traces.append(trace1)

    if colatitude_ref is not None and azimuth_ref is not None:

        x_ref = np.sin(colatitude_ref) * np.cos(azimuth_ref)
        y_ref = np.sin(colatitude_ref) * np.sin(azimuth_ref)
        z_ref = np.cos(colatitude_ref)

        if not hasattr(colatitude_ref, '__iter__'):
            colatitude_ref = [colatitude_ref]
            azimuth_ref = [azimuth_ref]
            x_ref = [x_ref]
            y_ref = [y_ref]
            z_ref = [z_ref]

        text_str2 = []
        for count, colatitude0 in enumerate(colatitude_ref):
            text_str2.append(
                u'({0:.2f}\N{DEGREE SIGN}, {1:.2f}\N{DEGREE SIGN})'.format(np.degrees(colatitude0),
                                                                           np.degrees(azimuth_ref[count]))
            )

        trace2 = go.Scatter3d(mode='markers', name='ground truth',
                              x=x_ref, y=y_ref, z=z_ref,
                              text=text_str2,
                              hoverinfo='name+text',
                              marker=dict(size=6, symbol='circle', opacity=0.6,
                                          line=dict(
                                              color='rgb(204, 204, 204)',
                                              width=2
                                          ),
                                          color='rgb(0, 0.447, 0.741)'))
        traces.append(trace2)


    if colatitude is not None and azimuth is not None:

        x_recon = np.sin(colatitude) * np.cos(azimuth)
        y_recon = np.sin(colatitude) * np.sin(azimuth)
        z_recon = np.cos(colatitude)

        if not hasattr(colatitude, '__iter__'):
            colatitude_ref = [colatitude]
            azimuth = [azimuth]
            x_recon = [x_recon]
            y_recon = [y_recon]
            z_recon = [z_recon]

        text_str3 = []
        for count, colatitude0 in enumerate(colatitude):
            text_str3.append(
                u'({0:.2f}\N{DEGREE SIGN}, {1:.2f}\N{DEGREE SIGN})'.format(np.degrees(colatitude0),
                                                                           np.degrees(azimuth[count]))
            )

        trace3 = go.Scatter3d(mode='markers', name='reconstruction',
                              x=x_recon, y=y_recon, z=z_recon,
                              text=text_str3,
                              hoverinfo='name+text',
                              marker=dict(size=6, symbol='diamond', opacity=0.6,
                                          line=dict(
                                              color='rgb(204, 204, 204)',
                                              width=2
                                          ),
                                          color='rgb(0.850, 0.325, 0.098)'))
        traces.append(trace3)

    data = go.Data(traces)

    layout = go.Layout(title='', autosize=False,
                       width=670, height=550, showlegend=True,
                       margin=go.Margin(l=45, r=45, b=55, t=45)
                       )

    layout['legend']['xanchor'] = 'center'
    layout['legend']['yanchor'] = 'top'
    layout['legend']['x'] = 0.5

    fig = go.Figure(data=data, layout=layout)
    plot(fig)


def sph_plot_diracs(
        colatitude_ref=None, azimuth_ref=None, colatitude=None, azimuth=None,
        dirty_img=None, colatitude_grid=None, azimuth_grid=None,
            
                    file_name='sph_recon_2d_dirac.pdf', **kwargs):
    '''
    This function plots the dirty image with sources locations on
    a flat projection of the sphere

    Parameters
    ----------
    colatitude_ref: ndarray, optional
        The colatitudes of a collection of reference points
    azimuths_ref: ndarray, optional
        The azimuths of a collection of reference points for the Diracs
    colatitude: ndarray, optional
        The colatitudes of the collection of points to visualize
    azimuth: ndarray, optional
        The azimuths of the collection of points to visualize
    dirty_img: ndarray
        A 2D map for displaying a pattern on the sphere under the points
    azimuth_grid: ndarray
        The azimuths indexing the dirty_img 2D map
    colatitude_grid: ndarray
        The colatitudes indexing the dirty_img 2D map
    '''

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings
        warnings.warn('Matplotlib is required for plotting')
        return

    fig = plt.figure(figsize=(6.47, 4), dpi=90)
    ax = fig.add_subplot(111, projection="mollweide")

    if dirty_img is not None and colatitude_grid is not None and azimuth_grid is not None:

        azimuth_plt_internal = azimuth_grid.copy()
        azimuth_plt_internal[azimuth_grid > np.pi] -= np.pi * 2
        p_hd = ax.pcolormesh(azimuth_plt_internal, np.pi / 2. - colatitude_grid,
                             np.real(dirty_img), cmap='Spectral_r',
                             linewidth=0, alpha=0.3,  # shading='gouraud',
                             antialiased=True, zorder=0)

        p_hd.set_edgecolor('None')
        p_hdc = fig.colorbar(p_hd, orientation='horizontal', use_gridspec=False,
                             anchor=(0.5, 1.2), shrink=0.75, spacing='proportional')
        # p_hdc.formatter.set_powerlimits((0, 0))
        p_hdc.ax.tick_params(labelsize=8.5)
        p_hdc.update_ticks()

    if colatitude_ref is not None and azimuth_ref is not None:

        if hasattr(colatitude_ref, '__iter__'):
            K = colatitude_ref.size
            x_ref = azimuth_ref.copy()
            y_ref = np.pi * 0.5 - colatitude_ref.copy()  # convert CO-LATITUDE to LATITUDE

            ind = x_ref > np.pi
            x_ref[ind] -= 2 * np.pi
        else:
            K = 1
            x_ref = azimuth_ref
            y_ref = np.pi * 0.5 - colatitude_ref

            if x_ref > np.pi:
                x_ref = x_ref - 2 * np.pi

        ax.scatter(x_ref, y_ref,
                   c=np.tile([0, 0.447, 0.741], (K, 1)),
                   s=70, alpha=0.75, zorder=11,
                   marker='^', linewidths=0, cmap='Spectral_r',
                   label='Original')


    if colatitude is not None and azimuth is not None:

        if hasattr(colatitude, '__iter__'):
            K_est = colatitude.size
            x = azimuth.copy()
            y = np.pi * 0.5 - colatitude.copy()  # convert CO-LATITUDE to LATITUDE

            ind = x > np.pi
            x[ind] -= 2 * np.pi  # scale conversion to -pi to pi
        else:
            K_est = 1
            x = azimuth
            y = np.pi * 0.5 - colatitude

            if x > np.pi:
                x = x - 2 * np.pi

        ax.scatter(x, y,
                   c=np.tile([0.850, 0.325, 0.098], (K_est, 1)),
                   s=100, alpha=0.75, zorder=12,
                   marker='*', linewidths=0, cmap='Spectral_r',
                   label='Reconstruction')

    ax.set_xticklabels([u'210\N{DEGREE SIGN}', u'240\N{DEGREE SIGN}',
                        u'270\N{DEGREE SIGN}', u'300\N{DEGREE SIGN}',
                        u'330\N{DEGREE SIGN}', u'0\N{DEGREE SIGN}',
                        u'30\N{DEGREE SIGN}', u'60\N{DEGREE SIGN}',
                        u'90\N{DEGREE SIGN}', u'120\N{DEGREE SIGN}',
                        u'150\N{DEGREE SIGN}'])

    ax.legend(scatterpoints=1, loc=8, fontsize=9,
              ncol=2, bbox_to_anchor=(0.5, -0.18),
              handletextpad=.2, columnspacing=1.7, labelspacing=0.1)  # framealpha=0.3,

    if 'title_str' in kwargs:
        ax.set_title(title_str, fontsize=11)

    ax.set_xlabel(r'azimuth', fontsize=11)
    ax.set_ylabel(r'latitude', fontsize=11)

    ax.xaxis.set_label_coords(0.5, 0.52)

    ax.grid(True)

    '''
    if save_fig:
        plt.savefig(file_name, format='pdf', dpi=300, transparent=True)
    '''

