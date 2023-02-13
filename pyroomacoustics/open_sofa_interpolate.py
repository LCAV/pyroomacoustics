import numpy as np
import scipy

try:
    import sofa

    has_sofa = True
except ImportError:
    has_sofa = False

import collections
import math
from timeit import default_timer as timer

from scipy.fft import fft, fftfreq, ifft
from scipy.signal import decimate
from scipy.spatial import KDTree, cKDTree, SphericalVoronoi
from scipy.interpolate import griddata
from .utilities import requires_matplotlib
from .doa import fibonnaci_spherical_sampling, spher2cart, cart2spher

# from numpy import linalg as LA
# from scipy.signal import find_peaks


def fibonacci_sphere(samples):
    """
    Creates a uniform fibonacci sphere.

    This version has some point at the pole which leads
    to a less *uniform* sampling. We should switch to
    fibonnaci_spherical_sampling

    Parameter
    ---------
    samples : (int)
        Points on the sphere

    Return
    --------
    Points : (np.array) shape(Points) = [no_of_points * 3]
        Cartesian coordinates of the points

    """

    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    i = np.arange(samples)
    y = 1 - i / float(samples - 1) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.column_stack([x, y, z])


def cal_sph_basis(theta, phi, degree, no_of_nodes):  # theta_target,phi_target
    """
     Calculate a spherical basis matrix

     Parameters
     -----------
     theta: array_like
        Phi (Azimuth) spherical coordinates of the SOFA file / fibonacci sphere
     phi: array_like
        Theta (Colatitude) spherical coordinates of the SOFA file / fibonacci sphere
     degree:(int)
        spherical harmonic degree
     no_of_nodes: (int)
        Length (theta)


     Return
    --------

    Ysh (np.array) shape(Ysh) = no_of_nodes * (degree + 1)**2
        Spherical harmonics basis matrix

    """

    Ysh = np.zeros((len(theta), (degree + 1) ** 2), dtype=np.complex_)

    ny0 = 1
    for j in range(no_of_nodes):
        for i in range(degree + 1):
            m = np.linspace(0, i, int((i - 0) / 1.0 + 1), endpoint=True, dtype=int)
            sph_vals = [
                scipy.special.sph_harm(order, i, theta[j], phi[j]) for order in m
            ]
            cal_index_Ysh_positive_order = (ny0 + m) - 1

            Ysh[j, cal_index_Ysh_positive_order] = sph_vals
            if i > 0:
                m_neg = np.linspace(
                    -i, -1, int((-1 - -i) / 1.0 + 1), endpoint=True, dtype=int
                )
                sph_vals_neg = [
                    scipy.special.sph_harm(order, i, theta[j], phi[j])
                    for order in m_neg
                ]
                cal_index_Ysh_negative_order = (ny0 + m_neg) - 1

                Ysh[j, cal_index_Ysh_negative_order] = sph_vals_neg

            # Update index for next degree
            ny0 = ny0 + 2 * i + 2
        ny0 = 1
    return Ysh


def calculation_pinv_voronoi_cells(Ysh, theta_16, no_of_lat):
    """
    Weighted least square solution "Analysis and Synthesis of Sound-Radiation with Spherical Arrays: Franz Zotter Page 76"

    Calculation of pseudo inverse and voronoi cells for regular sampling in spherical coordinates

    Parameters
    -----------
    Ysh: (np.ndarray)
        Spherical harmonic basis matrix
    theta_16: (int)
        Number of longitude on the original grid
    no_of_lat:
        Number of latitude on the original grid

    Returns:
    -------------------------------
    Ysh_tilda_inv : (np.ndarray)
        Weighted psuedo inverse of spherical harmonic basis matrix Ysh
    w_ : (np.ndarray)
        Weight on the original grid

    """

    res = (theta_16[:-1] + theta_16[1:]) / 2
    res = np.insert(res, len(res), np.pi)
    res = np.insert(res, 0, 0)

    w = -np.diff(np.cos(res))

    w_ = np.tile(w, no_of_lat)  # Repeat matrice n times
    # w_ /= w_.sum()  # normalize

    # the following is equivalent to multiplication by diagonal matrix of weights
    Ysh_tilda = w_[:, None] * Ysh

    Ysh_tilda_inv = np.linalg.pinv(
        Ysh_tilda, rcond=1e-2
    )  # rcond is inverse of the condition number

    return Ysh_tilda_inv, w_


def calculation_pinv_voronoi_cells_general(Ysh, points):
    """
    Weighted least square solution "Analysis and Synthesis of Sound-Radiation with
    Spherical Arrays: Franz Zotter Page 76"

    Calculation of pseudo inverse and voronoi cells for arbitrary sampling of the sphere.

    Parameters
    -----------
    Ysh: (np.ndarray)
        Spherical harmonic basis matrix
    points: numpy.ndarray, (n_points, 3)
        The sampling points on the sphere

    Returns:
    -------------------------------
    Ysh_tilda_inv : (np.ndarray)
        Weighted pseudo inverse of spherical harmonic basis matrix Ysh
    w_ : (np.ndarray)
        Weight on the original grid
    """

    # The weights are the areas of the voronoi cells
    sv = SphericalVoronoi(points)
    w_ = sv.calculate_areas()
    w_ /= w_.sum()  # normalize

    Ysh_tilda_inv = np.linalg.pinv(
        w_[:, None] * Ysh, rcond=1e-2
    )  # rcond is inverse of the condition number

    return Ysh_tilda_inv, w_


def DIRPAT_pattern_enum_id(DIRPAT_pattern_enum, source=False):
    """
    Assigns DIRPAT pattern enum to respective id present in the DIRPAT SOFA files.
    Works only for mic and source files


    """

    if source is not True:
        if "AKG_c480" in DIRPAT_pattern_enum:
            id = 0
        elif "AKG_c414K" in DIRPAT_pattern_enum:
            id = 1
        elif "AKG_c414N" in DIRPAT_pattern_enum:
            id = 2
        elif "AKG_c414S" in DIRPAT_pattern_enum:
            id = 3
        elif "AKG_c414A" in DIRPAT_pattern_enum:
            id = 4
        elif "EM_32" in DIRPAT_pattern_enum:
            id = int(DIRPAT_pattern_enum.split("_")[-1])
        else:
            raise ValueError("Please specifiy correct DIRPAT_pattern_enum for mic")
    else:
        if "Genelec_8020" in DIRPAT_pattern_enum:
            id = 0
        elif "Lambda_labs_CX-1A" in DIRPAT_pattern_enum:
            id = 1
        elif "HATS_4128C" in DIRPAT_pattern_enum:
            id = 2
        elif "Tannoy_System_1200" in DIRPAT_pattern_enum:
            id = 3
        elif "Neumann_KH120A" in DIRPAT_pattern_enum:
            id = 4
        elif "Yamaha_DXR8" in DIRPAT_pattern_enum:
            id = 5
        elif "BM_1x12inch_driver_closed_cabinet" in DIRPAT_pattern_enum:
            id = 6
        elif "BM_1x12inch_driver_open_cabinet" in DIRPAT_pattern_enum:
            id = 7
        elif "BM_open_stacked_on_closed_withCrossoverNetwork" in DIRPAT_pattern_enum:
            id = 8
        elif "BM_open_stacked_on_closed_fullrange" in DIRPAT_pattern_enum:
            id = 9
        elif "Palmer_1x12inch" in DIRPAT_pattern_enum:
            id = 10
        elif "Vibrolux_2x10inch" in DIRPAT_pattern_enum:
            id = 11
        else:
            raise ValueError("Please specifiy correct DIRPAT_pattern_enum for source")

    return id


class DIRPATInterpolate:

    """
    Intialization class performs the following functions
    Open DIRPAT Files
    Interpolate on fibonacci sphere
    Rotate the pattern w.r.t to the given azimuth(phi) [-pi, pi] and colatitude(theta) [0,pi] defines the orientation of the source or a receiver
    Reutrns response in the given direction w.r.t to the given angle of arrivals(microphones) / angle of departures(angle of departures)

    Parameters
    --------------
    path : (string)
        Path towards the specific DIRPAT file
    fs  : (int)
        Sampling frequency of the microphone or source FIR's filters , should be  <= simulation frequency
    DIRPAT_pattern_enum :  (string)
        The specific pattern in the DIRPAT files are associated with id's , presented in the github document
    source  : (Boolean)
        If the DIRPAT files represnts one of source or receiver's
    interpolate  : (Boolean)
        Interpolate the FIR filter or not
    no_of_points_fibo_sphere  : (int)
        How many points should be there on fibonnacci sphere
    azimuth_simulation  , colatitude_simulation : (int)
        Orientation of the source/mic in the simulation environment in degrees.

    """

    def __init__(
        self,
        path,
        fs,
        DIRPAT_pattern_enum=None,
        source=False,
        interpolate=True,
        no_of_points_fibo_sphere=1000,
        azimuth_simulation=0,
        colatitude_simulation=0,
    ):
        self.path = path
        self.id = DIRPAT_pattern_enum_id(DIRPAT_pattern_enum, source=source)
        self.source = source
        self.DIRPAT_pattern_enum = DIRPAT_pattern_enum

        self.fs = fs

        (
            self.phi_sofa,  # azimuth
            self.theta_sofa,  # colatitude
            self.r_sofa,
            self.theta_16,
            self.sofa_msr_fir,
            self.no_of_nodes,
            self.samples_size_ir,
            self.no_of_lat,
        ) = self.open_sofa_database(path=self.path, fs=self.fs)

        self.sofa_x, self.sofa_y, self.sofa_z = spher2cart(
            self.phi_sofa, self.theta_sofa, self.r_sofa
        )

        self.interpolate = interpolate

        # Rotation matrix

        n_c = colatitude_simulation
        n_a = azimuth_simulation
        R_y = np.array(
            [[np.cos(n_c), 0, np.sin(n_c)], [0, 1, 0], [-np.sin(n_c), 0, np.cos(n_c)]]
        )
        R_z = np.array(
            [[np.cos(n_a), -np.sin(n_a), 0], [np.sin(n_a), np.cos(n_a), 0], [0, 0, 1]]
        )
        res = np.matmul(R_z, R_y)

        if interpolate:
            self.degree = 12

            self.points = np.array(fibonacci_sphere(samples=no_of_points_fibo_sphere))
            # self.points = fibonnaci_spherical_sampling(no_of_points_fibo_sphere).T
            self.phi_fibo, self.theta_fibo, self.r_fibo = cart2spher(self.points.T)

            # All the computations are in radians for phi = range (0, 2*np.pi) and theta = range (0, np.pi)
            # Just for numpy, the function accepts phi and theta in the opposite way

            self.theta_sofa_np = self.phi_sofa
            self.phi_sofa_np = self.theta_sofa
            self.Ysh = cal_sph_basis(
                self.theta_sofa_np, self.phi_sofa_np, self.degree, self.no_of_nodes
            )

            # Calculate spherical basis for the target grid (fibonacci grid)
            self.theta_fibo_np = self.phi_fibo
            self.phi_fibo_np = self.theta_fibo

            self.Ysh_fibo = cal_sph_basis(
                self.theta_fibo_np,
                self.phi_fibo_np,
                self.degree,
                no_of_points_fibo_sphere,
            )

            # calculate pinv and voronoi cells for least square solution for the whole grid

            self.Ysh_tilda_inv, self.w_ = calculation_pinv_voronoi_cells(
                self.Ysh, self.theta_16, self.no_of_lat
            )
            """
            points = np.array([self.sofa_x, self.sofa_y, self.sofa_z]).T
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            self.Ysh_tilda_inv, self.w_ = calculation_pinv_voronoi_cells_general(
                self.Ysh, points
            )
            """

            self.freq_angles_fft = np.zeros(
                (self.no_of_nodes, self.samples_size_ir), dtype=np.complex_
            )  # N-point FFT
            self.sh_coeffs_expanded_original_grid = 0
            self.sh_coeffs_expanded_target_grid = 0

            # Rotate the target grid (Fibonacci grid)

            self.rotated_fibo_points = np.matmul(res, self.points.T)

            (
                self.rotated_fibo_phi,
                self.rotated_fibo_theta,
                self.rotated_fibo_r,
            ) = cart2spher(self.rotated_fibo_points)
            self.nn_kd_tree_rotated_fibo_grid = cKDTree(
                np.hstack(
                    (
                        self.rotated_fibo_phi.reshape(-1, 1),
                        self.rotated_fibo_theta.reshape(-1, 1),
                    )
                )
            )

            self.cal_spherical_coeffs_grid_and_interpolate()

        else:
            self.freq_angles_fft = np.zeros(
                (self.no_of_nodes, self.samples_size_ir), dtype=np.complex_
            )
            for i in range(self.no_of_nodes):
                self.freq_angles_fft[i, :] = fft(self.sofa_msr_fir[i, :])

            cart_points = np.empty((3, self.phi_sofa.shape[0]))
            cart_points[0, :] = self.sofa_x
            cart_points[1, :] = self.sofa_y
            cart_points[2, :] = self.sofa_z

            self.rotated_sofa_points = np.matmul(res, cart_points)

            (
                self.rotated_sofa_phi,
                self.rotated_sofa_theta,
                self.rotated_sofa_r,
            ) = cart2spher(self.rotated_sofa_points)

            # print(np.max(self.rotated_sofa_phi), np.min(self.rotated_sofa_phi))
            self.nn_kd_tree_rotated_sofa_grid = cKDTree(
                np.hstack(
                    (
                        self.rotated_sofa_phi.reshape(-1, 1),
                        self.rotated_sofa_theta.reshape(-1, 1),
                    )
                )
            )

    def cal_spherical_coeffs_grid_and_interpolate(self):
        """
        Weighted least square solution to calculate discrete sphreical harmonic coeffs , Using the spherical harmonic coeffs
        interpolation is done on fibonacci sphere
        """

        # freq_angles_fft = np.zeros((self.no_of_nodes, self.samples_size_ir// 2), dtype=np.complex_)

        # Take n-point dft of the FIR's on the grid
        for i in range(self.no_of_nodes):
            self.freq_angles_fft[i, :] = fft(
                self.sofa_msr_fir[i, :]
            )  # [:(self.samples_size_ir // 2)] #self.samples_size_ir//2

        g_tilda = self.w_[:, None] * self.freq_angles_fft

        # Shape w_ : (540*540) or (480*480)
        # Shape g_tilda : (480*256),(480*128),(540*2048),(540*1048),(540,683)

        gamma_full_scale = np.matmul(
            self.Ysh_tilda_inv, g_tilda
        )  # Coeffs for every freq band , for all the nodes present in the sphere

        """
        #Shape gamma_full_scale : ((deg+1)^2 * 256 | 128 | 2048 | 1024 | 683) Coeffs for every frequency bin

        #Select coeffs for particular frequency for plotting purpose
        #For sources select between (0-1023), receivers (0-128)

        #freq_bin=20
        #gamma = gamma_full_scale[:, freq_bin] #[:(self.samples_size_ir//2)]

        #Shape gamma : ((deg+1)^2) Coeffs for a particular frequency bin .
        """

        self.sh_coeffs_expanded_original_grid = np.fft.ifft(
            np.matmul(self.Ysh, gamma_full_scale), axis=-1
        )
        self.sh_coeffs_expanded_target_grid = np.fft.ifft(
            np.matmul(self.Ysh_fibo, gamma_full_scale), axis=-1
        )

    def open_sofa_database(self, path, fs=16000):
        # Open DirPat database
        if not has_sofa:
            raise ValueError(
                "The package 'sofa' needs to be installed to call this function. Install by doing `pip install sofa`"
            )

        if self.source == True:
            file_sofa = sofa.Database.open(path)

            # Receiver positions

            rcv_pos = file_sofa.Receiver.Position.get_values()

            # CHEAP HACCCKK SPECIFICALLY FOR DIRPAT

            rcv_pos_RS = np.reshape(rcv_pos, [36, 15, 3])

            rcv_pos = np.swapaxes(rcv_pos_RS, 0, 1).reshape([540, -1])

            ###########################

            # Get impulse responses from all the measurements

            IR_S = file_sofa.Data.IR.get_values()

            # Look for source of specific type requested by user"

            rcv_msr = IR_S[
                self.id, :, :
            ]  # First receiver #Shape ( no_sources * no_measurement_points * no_samples_IR)

            # downsample the fir filter.

            rcv_msr = decimate(
                rcv_msr,
                int(round(file_sofa.Data.SamplingRate.get_values()[0] / fs)),
                axis=-1,
            )

            no_of_nodes = 540

            # samples = file_sofa.Dimensions.N  # Samples per IR
            samples = rcv_msr.shape[
                1
            ]  # Number of samples changed after downsampling the FIR filter

            # All measurements should be in = radians phi [0,2*np.pi] , theta [0,np.pi]

            phi_rcv = rcv_pos[:, 0]
            theta_rcv = rcv_pos[:, 1]
            r_rcv = rcv_pos[:, 2]

            # Calculate no of latitudes and longitudes in the grid

            no_of_lat = list(collections.Counter(theta_rcv).values())[0]

            # no_of_long = len(no_of_lat)

            theta_16 = np.array(
                [theta_rcv[i] for i in range(len(theta_rcv)) if i % no_of_lat == 0]
            )

            return (
                phi_rcv,
                theta_rcv,
                r_rcv,
                theta_16,
                rcv_msr,
                no_of_nodes,
                samples,
                no_of_lat,
            )
        else:
            file_sofa = sofa.Database.open(path)
            # Source positions

            src_pos = file_sofa.Source.Position.get_values()

            if "EM_32" in self.DIRPAT_pattern_enum:
                src_pos = np.deg2rad(src_pos)

            else:
                # CHEAP HACCCKK SPECIFICALLY FOR DIRPAT

                src_pos_RS = np.reshape(src_pos, [30, 16, 3])

                src_pos = np.swapaxes(src_pos_RS, 0, 1).reshape([480, -1])

                ###########################

            # Get impulse responses from all the measurements

            IR_S = file_sofa.Data.IR.get_values()

            # Look for receiver of specific type requested by user"

            rcv_msr = IR_S[
                :, self.id, :
            ]  # First receiver #Shape (no_measurement_points * no_receivers * no_samples_IR)

            rcv_msr = decimate(
                rcv_msr,
                int(round(file_sofa.Data.SamplingRate.get_values()[0] / fs)),
                axis=-1,
            )

            no_of_nodes = file_sofa.Dimensions.M  # Number of measurement points

            # samples = file_sofa.Dimensions.N  # Samples per IR
            samples = rcv_msr.shape[1]

            # All measurements should be in = radians phi [0,2*np.pi] , theta [0,np.pi]

            phi_src = src_pos[:, 0]
            theta_src = src_pos[:, 1]
            r_src = src_pos[:, 2]

            no_of_lat = list(collections.Counter(theta_src).values())[0]

            # no_of_long = len(no_of_lat)

            theta_16 = np.array(
                [theta_src[i] for i in range(len(theta_src)) if i % no_of_lat == 0]
            )

            return (
                phi_src,
                theta_src,
                r_src,
                theta_16,
                rcv_msr,
                no_of_nodes,
                samples,
                no_of_lat,
            )

    def cal_index_knn(self, azimuth, colatitude):
        """
        Calculates nearest neighbour for all the image source for the source and the receiver
        This method exist so that we don't have to query each image source , nearest neighbour for all the incoming angles (mic) and departure angles(sources)
        are calculated at once.
        Parameters
        ----------

        azimuth: (np.ndarray)
             list of azimuth angles [-pi,pi]
        colatitude: (np.ndarray)
            list of colatitude angles [0,pi]

        Return : (np.ndarray)
        -------
          Return nearest index on the grid based on the input angles using kDtree
        """

        longitude = azimuth.reshape(-1, 1)
        latitude = colatitude.reshape(-1, 1)

        start = timer()
        if self.interpolate:
            dd, ii = self.nn_kd_tree_rotated_fibo_grid.query(
                np.hstack((longitude, latitude))
            )
        else:
            dd, ii = self.nn_kd_tree_rotated_sofa_grid.query(
                np.hstack((longitude, latitude))
            )  # Query on the rotated set of points
        end = timer()
        # print("Time taken For Query Once ", end - start)
        return ii

    def neareast_neighbour(self, index):
        """
        Retrives the FIR in frequency resolution according to the given index parameter. Index is found using query on kdtree based on incoming or outgoing angles of the image sources

        Parameters
        -------------------------
        index : (np.ndarray)
            Index of the points on the grid

        :Return (np.ndarray) [Coeffs of DFT]
        -------------------------
            FIR in frequency resolution for the particular query angle.

        """

        if self.interpolate:
            # dd, ii = self.nn_kd_tree_rotated_fibo_grid.query([longitude, latitude])  #Query on the rotated set of points
            return self.sh_coeffs_expanded_target_grid[index, :]
        else:
            # dd, ii = self.nn_kd_tree_rotated_sofa_grid.query([longitude, latitude])  # Query on the rotated set of points
            return self.sofa_msr_fir[index, :]

    def change_orientation(self, azimuth_change, colatitude_change):
        """
        Change of rotation without repeating the whole process only works for the sofa file where we are doing interpolation on fibo sphere
        Made for IWAENC.

        """

        n_c = np.radians(colatitude_change)
        n_a = np.radians(azimuth_change)
        R_y = np.array(
            [[np.cos(n_c), 0, np.sin(n_c)], [0, 1, 0], [-np.sin(n_c), 0, np.cos(n_c)]]
        )
        R_z = np.array(
            [[np.cos(n_a), -np.sin(n_a), 0], [np.sin(n_a), np.cos(n_a), 0], [0, 0, 1]]
        )
        res = np.matmul(R_z, R_y)

        self.rotated_fibo_points = np.matmul(res, self.points.T)

        (
            self.rotated_fibo_phi,
            self.rotated_fibo_theta,
            self.rotated_fibo_r,
        ) = cart2spher(self.rotated_fibo_points)

        self.nn_kd_tree_rotated_fibo_grid = cKDTree(
            np.hstack(
                (
                    self.rotated_fibo_phi.reshape(-1, 1),
                    self.rotated_fibo_theta.reshape(-1, 1),
                )
            )
        )

    def plot(self, freq_bin=0):
        """
        Plot 3D plots of the SOFA file , interpolated fibonnacci sphere and roatated fibo and sofa spheres.

        Display gain for a specified freq bin


        :Parameter (int) freq_bin:Bin number

        """
        import matplotlib.pyplot as plt

        fg = plt.figure(figsize=plt.figaspect(0.5))

        if self.interpolate:
            ax1 = fg.add_subplot(1, 4, 1, projection="3d")
            ax_ = ax1.scatter(
                self.sofa_x,
                self.sofa_y,
                self.sofa_z,
                c=np.abs(self.freq_angles_fft[:, freq_bin]),
                s=100,
            )
            ax1.set_title("Mag|STFT| of one of the SOFA Files")
            fg.colorbar(ax_, shrink=0.5, aspect=5)

            ax2 = fg.add_subplot(1, 4, 2, projection="3d")
            ax_1 = ax2.scatter(
                self.sofa_x,
                self.sofa_y,
                self.sofa_z,
                c=np.abs(
                    fft(self.sh_coeffs_expanded_original_grid, axis=-1)[:, freq_bin]
                ),
                s=100,
            )
            ax2.set_title("Expanded Initial Grid")
            fg.colorbar(ax_1, shrink=0.5, aspect=5)

            ax3 = fg.add_subplot(1, 4, 4, projection="3d")
            ax_2 = ax3.scatter(
                self.rotated_fibo_points[0, :],
                self.rotated_fibo_points[1, :],
                self.rotated_fibo_points[2, :],
                c=np.abs(
                    fft(self.sh_coeffs_expanded_target_grid, axis=-1)[:, freq_bin]
                ),
                s=50,
            )
            ax3.set_title(
                "Interpolated And Rotated Target Grid Based On Analytic SOFA Receiver"
            )
            fg.colorbar(ax_2, shrink=0.5, aspect=5)

            ax4 = fg.add_subplot(1, 4, 3, projection="3d")
            ax_3 = ax4.scatter(
                self.points[:, 0],
                self.points[:, 1],
                self.points[:, 2],
                c=np.abs(
                    fft(self.sh_coeffs_expanded_target_grid, axis=-1)[:, freq_bin]
                ),
                s=50,
            )  # fibo_debug
            ax4.set_title("Interpolated Fibo Grid")
            fg.colorbar(ax_3, shrink=0.5, aspect=5)
        else:
            ax1 = fg.add_subplot(1, 2, 1, projection="3d")
            ax_ = ax1.scatter(
                self.sofa_x,
                self.sofa_y,
                self.sofa_z,
                c=np.abs(self.freq_angles_fft[:, freq_bin]),
                s=100,
            )
            ax1.set_title("Mag|STFT| of one of the SOFA Files")
            fg.colorbar(ax_, shrink=0.5, aspect=5)

            ax2 = fg.add_subplot(1, 2, 2, projection="3d")
            ax_1 = ax2.scatter(
                self.rotated_sofa_points[0, :],
                self.rotated_sofa_points[1, :],
                self.rotated_sofa_points[2, :],
                c=np.abs(self.freq_angles_fft[:, freq_bin]),
                s=100,
            )
            ax2.set_title("Expanded Initial Grid")
            fg.colorbar(ax_1, shrink=0.5, aspect=5)

        fg.tight_layout(pad=3.0)
        plt.show()

    @requires_matplotlib
    def plot_new(self, freq_bin=0, n_grid=100, ax=None, depth=False, offset=None):
        import matplotlib.pyplot as plt
        from matplotlib import cm

        if self.interpolate:
            cart = self.points
            length = np.abs(
                fft(self.sh_coeffs_expanded_target_grid, axis=-1)[:, freq_bin]
            )
        else:
            cart = np.array([self.sofa_x, self.sofa_y, self.sofa_z]).T
            length = np.abs(self.freq_angles_fft[:, freq_bin])

        # regrid the data on a 2D grid
        g = np.linspace(-1, 1, n_grid)
        AZ, COL = np.meshgrid(
            np.linspace(0, 2 * np.pi, n_grid), np.linspace(0, np.pi, n_grid // 2)
        )
        # multiply by 0.99 to make sure the interpolation grid is inside the convex hull
        # of the original points, otherwise griddata returns NaN
        X = np.cos(AZ) * np.sin(COL) * 0.99
        Y = np.sin(AZ) * np.sin(COL) * 0.99
        Z = np.cos(COL) * 0.99
        grid = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        interp_len = griddata(cart, length, grid, method="linear")
        V = interp_len.reshape((n_grid // 2, n_grid))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")

        # Colour the plotted surface according to the sign of Y.
        cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("coolwarm"))
        cmap.set_clim(0, V.max())

        if depth:
            X *= V
            Y *= V
            Z *= V

        surf = ax.plot_surface(
            X, Y, Z, facecolors=cmap.to_rgba(V), linewidth=0, antialiased=False
        )

        plt.show()

        return ax
