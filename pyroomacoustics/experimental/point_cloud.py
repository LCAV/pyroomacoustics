'''
Point Clouds
============
Contains PointCloud class.

Given a number of points and their relative distances, this class aims at
reconstructing their relative coordinates.
'''

from __future__ import division, print_function

# Provided by LCAV
import numpy as np
from scipy import linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class PointCloud:

    def __init__(self, m=1, dim=3, diameter=0., X=None, labels=None, EDM=None):
        '''
        Parameters
        ----------

        m : int, optional
            Number of markers
        diameter : float, optional
            Diameter of the marker points (added to distance measurements)
        dim : int, optional
            Dimension of ambient space (default 3)
        X : ndarray, optional
            Array of column vectors with locations of markers
        '''

        # set the marker diameter
        self.diameter = diameter

        if EDM is not None:
            self.dim = dim
            self.fromEDM(EDM, labels=labels)

        elif X is not None:
            self.m = X.shape[1]
            self.dim = X.shape[0]
            self.X = X

            if labels is None:
                self.labels = [str(i) for i in range(self.m)]

        else:
            self.m = m
            self.dim = dim
            self.X = np.zeros((self.dim, self.m))

        # Now set the labels
        if labels is not None:
            if len(labels) == self.m:
                self.labels = labels
            else:
                raise ValueError('There needs to be one label per marker point')
        else:
            self.labels = [str(i) for i in range(self.m)]


    def __getitem__(self,ref):

        if isinstance(ref, (str, unicode)):
            if self.labels is None:
                raise ValueError('Labels not set for this marker set')
            index = self.labels.index(ref)
        elif isinstance(ref, int) or isinstance(ref, slice):
            index = ref
        elif isinstance(ref, list):
            index = [self.labels.index(s) if isinstance(s, (str, unicode)) else s for s in ref]
        else:
            index = int(ref)

        return self.X[:,index]

    def copy(self):
        ''' Return a deep copy of this marker set object '''

        new_marker = PointCloud(X=self.X.copy(), labels=self.labels, diameter=self.diameter)
        return new_marker

    def key2ind(self, ref):
        ''' Get the index location from a label '''

        if isinstance(ref, (str, unicode)):
            if self.labels is None:
                raise ValueError('Labels must be defined to be used to access markers')
            else:
                return self.labels.index(ref)
        else:
            return int(ref)

    def fromEDM(self, D, labels=None, method='mds'):
        '''
        Compute the position of markers from their Euclidean Distance Matrix

        Parameters
        ----------
        D: square 2D ndarray
            Euclidean Distance Matrix (matrix containing squared distances between points
        labels: list, optional
            A list of human friendly labels for the markers (e.g. 'east', 'west', etc)
        method: str, optional 
            The method to use
            * 'mds' for multidimensional scaling (default)
            * 'tri' for trilateration
        '''

        if D.shape[0] != D.shape[1]:
            raise ValueError('The distance matrix must be square')

        self.m = D.shape[0]

        if method == 'tri':
            self.trilateration(D)
        else:
            self.classical_mds(D)

    def classical_mds(self, D):
        ''' 
        Classical multidimensional scaling

        Parameters
        ----------
        D : square 2D ndarray
            Euclidean Distance Matrix (matrix containing squared distances between points
        '''

        # Apply MDS algorithm for denoising
        n = D.shape[0]
        J = np.eye(n) - np.ones((n,n))/float(n)
        G = -0.5*np.dot(J, np.dot(D, J))

        s, U = np.linalg.eig(G)

        # we need to sort the eigenvalues in decreasing order
        s = np.real(s)
        o = np.argsort(s)
        s = s[o[::-1]]
        U = U[:,o[::-1]]

        S = np.diag(s)[0:self.dim,:]
        self.X = np.dot(np.sqrt(S),U.T)

    def trilateration_single_point(self, c, Dx, Dy):
        '''
        Given x at origin (0,0) and y at (0,c) the distances from a point
        at unknown location Dx, Dy to x, y, respectively, finds the position of the point.
        '''

        z = (c**2 - (Dy**2 - Dx**2)) / (2*c)
        t = np.sqrt(Dx**2 - z**2)

        return np.array([t,z])

    def trilateration(self, D):
        '''
        Find the location of points based on their distance matrix using trilateration

        Parameters
        ----------
        D : square 2D ndarray
            Euclidean Distance Matrix (matrix containing squared distances between points
        '''

        dist = np.sqrt(D)

        # Simpler algorithm (no denoising)
        self.X = np.zeros((self.dim, self.m))

        self.X[:,1] = np.array([0, dist[0,1]])
        for i in range(2,m):
            self.X[:,i] = self.trilateration_single_point(self.X[1,1],
                    dist[0,i], dist[1,i])

    def EDM(self):
        ''' Computes the EDM corresponding to the marker set '''
        if self.X is None:
            raise ValueError('No marker set')

        G = np.dot(self.X.T, self.X)
        return np.outer(np.ones(self.m), np.diag(G)) \
            - 2*G + np.outer(np.diag(G), np.ones(self.m))

    def normalize(self, refs=None):
        '''
        Reposition points such that x0 is at origin, x1 lies on c-axis
        and x2 lies above x-axis, keeping the relative position to each other.
        The z-axis is defined according to right hand rule by default.

        Parameters
        ----------
        refs : list of 3 ints or str
            The index or label of three markers used to define (origin, x-axis, y-axis)
        left_hand : bool, optional (default False)
            Normally the z-axis is defined using right-hand rule, this flag allows to override this behavior
        '''

        if refs is None:
            refs = [0,1,2,3]

        # Transform references to indices if needed
        refs = [self.key2ind(s) for s in refs]

        if self.dim == 2 and len(refs) < 3:
            raise ValueError('In 2D three reference points are needed to define a reference frame')
        elif self.dim == 3 and len(refs) < 4:
            raise ValueError('In 3D four reference points are needed to define a reference frame')

        # set first point to origin
        X0 = self.X[:,refs[0],None]
        Y = self.X - X0

        # Rotate around z to align x-axis to second point
        theta = np.arctan2(Y[1,refs[1]],Y[0,refs[1]])
        c = np.cos(theta)
        s = np.sin(theta)
        if self.dim == 2:
            H = np.array([[c, s],[-s, c]])
        elif self.dim == 3:
            H = np.array([[c, s, 0],[-s, c, 0], [0, 0, 1]])
        Y = np.dot(H,Y)

        if self.dim == 2:
            # set third point to lie above x-axis
            if Y[1,refs[2]] < 0:
                Y[1,:] *= -1

        elif self.dim == 3:
            # In 3D we also want to make sur z-axis points up
            theta = np.arctan2(Y[2,refs[2]],Y[1,refs[2]])
            c = np.cos(theta)
            s = np.sin(theta)
            H = np.array([[1, 0, 0], [0, c, s],[0, -s, c]])
            Y = np.dot(H,Y)

        # Flip the z-axis if requested
        if self.dim == 3 and Y[2,refs[3]] < 0:
            Y[2,:] *= -1

        self.X = Y

    def center(self, marker):
        ''' Translate the marker set so that the argument is the origin. '''

        index = self.key2ind(marker)
        self.X -= self.X[:,index,None]

    def align(self, marker, axis):
        '''
        Rotate the marker set around the given axis until it is aligned onto the given marker

        Parameters
        ----------
        marker : int or str
            the index or label of the marker onto which to align the set
        axis : int
            the axis around which the rotation happens
        '''

        index = self.key2ind(marker)
        axis = ['x','y','z'].index(axis) if isinstance(marker, (str, unicode)) else axis

        # swap the axis around which to rotate to last position
        Y = self.X
        if self.dim == 3:
            Y[axis,:], Y[2,:] = Y[2,:], Y[axis,:]

        # Rotate around z to align x-axis to second point
        theta = np.arctan2(Y[1,index],Y[0,index])
        c = np.cos(theta)
        s = np.sin(theta)
        H = np.array([[c, s],[-s, c]])
        Y[:2,:] = np.dot(H,Y[:2,:])

        if self.dim == 3:
            Y[axis,:], Y[2,:] = Y[2,:], Y[axis,:]

    def flatten(self, ind):
        '''
        Transform the set of points so that the subset of markers given as argument is
        as close as flat (wrt z-axis) as possible.

        Parameters
        ----------
        ind : list of bools
            Lists of marker indices that should be all in the same subspace
        '''

        # Transform references to indices if needed
        ind = [self.key2ind(s) for s in ind]

        # center point cloud around the group of indices
        centroid = self.X[:,ind].mean(axis=1, keepdims=True)
        X_centered = self.X - centroid

        # The rotation is given by left matrix of SVD
        U,S,V = la.svd(X_centered[:,ind], full_matrices=False)

        self.X = np.dot(U.T, X_centered) + centroid

    def correct(self, corr_dic):
        ''' correct a marker location by a given vector '''

        for key, val in corr_dic.items():
            ind = self.key2ind(key)
            self.X[:,ind] += val

    def doa(self, receiver, source):
        ''' Computes the direction of arrival wrt a source and receiver '''

        s_ind = self.key2ind(source)
        r_ind = self.key2ind(receiver)

        # vector from receiver to source
        v = self.X[:,s_ind] - self.X[:,r_ind]

        azimuth = np.arctan2(v[1], v[0])
        elevation = np.arctan2(v[2], la.norm(v[:2]))

        return np.array([azimuth, elevation])

    def plot(self, axes=None, show_labels=True, **kwargs):

        if self.dim == 2:

            # Create a figure if needed
            if axes is None:
                axes = plt.subplot(111)

            axes.plot(self.X[0,:], self.X[1,:], **kwargs)
            axes.axis(aspect='equal')
            plt.show()


        elif self.dim == 3:
            if axes is None:
                fig = plt.figure()
                axes = fig.add_subplot(111, projection='3d')
            axes.scatter(self.X[0,:], self.X[1,:], self.X[2,:], **kwargs)
             
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel('Z')
            plt.show()

        if show_labels and self.labels is not None:
            eps = np.linalg.norm(self.X[:,0] - self.X[:,1])/100
            for i in range(self.m):
                if self.dim == 2:
                    axes.text(self.X[0,i]+eps, self.X[1,i]+eps, self.labels[i])
                elif self.dim == 3:
                    axes.text(self.X[0,i]+eps, self.X[1,i]+eps, self.X[2,i]+eps, self.labels[i], None)

        return axes




if __name__ == '__main__':

    # number of markers
    m = 4
    dim = 2

    D = np.zeros((m,m))

    marker_diameter = 0.040 # 4 cm

    M_orig = MarkerSet(X=np.array([[0.,0.],[0.7,0.],[0.7,0.7],[0.,0.7]]).T)
    D = np.sqrt(M_orig.EDM())

    """
    D[0,1] = D[1,0] = 4.126 + marker_diameter
    D[0,2] = D[2,0] = 6.878 + marker_diameter
    D[0,3] = D[3,0] = 4.508 + marker_diameter
    D[1,2] = D[2,1] = 4.401 + marker_diameter
    D[1,3] = D[3,1] = 7.113 + marker_diameter
    D[3,2] = D[2,3] = 7.002 + marker_diameter
    """

    M1 = MarkerSet(m=m, dim=dim, diameter=marker_diameter)
    M1.fromEDM(D**2)
    M1.normalize()

    M2 = MarkerSet(m=m, dim=dim, diameter=marker_diameter)
    M2.fromEDM(D**2, method='tri')
    M2.normalize()

    M2.plot(marker='ko', labels=True)
    M1.plot(marker='rx')
    plt.show()

