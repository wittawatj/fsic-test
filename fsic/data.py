__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import math
import matplotlib.pyplot as plt
import numpy as np
import fsic.util as util
import matplotlib.pyplot as plt
import scipy.stats as stats

class PairedData(object):
    """Class representing paired data for independence testing

    properties:
    X, Y: numpy array. X and Y are paired of the same sample size. The
        dimensions are not necessarily the same.
    """

    def __init__(self, X, Y, label=None):
        """
        :param X: n x d numpy array for dataset X
        :param Y: n x d' numpy array for dataset Y
        """
        self.X = X
        self.Y = Y
        # short description to be used as a plot label
        self.label = label

        nx, dx = X.shape
        ny, dy = Y.shape
        if nx != ny:
            raise ValueError('Data size of the paired sample must be the same.')

        if not np.all(np.isfinite(X)):
            print('X:')
            print(util.fullprint(X))
            raise ValueError('Not all elements in X are finite.')

        if not np.all(np.isfinite(Y)):
            print('Y:')
            print(util.fullprint(Y))
            raise ValueError('Not all elements in Y are finite.')

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0) 
        mean_y = np.mean(self.Y, 0)
        std_y = np.std(self.Y, 0) 
        prec = 4
        desc = ''
        desc += 'E[x] = %s \n'%(np.array_str(mean_x, precision=prec ) )
        desc += 'E[y] = %s \n'%(np.array_str(mean_y, precision=prec ) )
        desc += 'Std[x] = %s \n' %(np.array_str(std_x, precision=prec))
        desc += 'Std[y] = %s \n' %(np.array_str(std_y, precision=prec))
        return desc

    def dx(self):
        """Return the dimension of X."""
        dx = self.X.shape[1]
        return dx

    def dy(self):
        """Return the dimension of Y."""
        dy = self.Y.shape[1]
        return dy

    def sample_size(self):
        return self.X.shape[0]

    def xy(self):
        """Return (X, Y) as a tuple"""
        return (self.X, self.Y)

    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. Assume n is the same 
        for both X, Y. 
        
        Return (PairedData for tr, PairedData for te)"""
        X = self.X
        Y = self.Y
        nx, dx = X.shape
        ny, dy = Y.shape
        if nx != ny:
            raise ValueError('Require nx = ny')
        Itr, Ite = util.tr_te_indices(nx, tr_proportion, seed)
        label = '' if self.label is None else self.label
        tr_data = PairedData(X[Itr, :], Y[Itr, :], 'tr_' + label)
        te_data = PairedData(X[Ite, :], Y[Ite, :], 'te_' + label)
        return (tr_data, te_data)

    def subsample(self, n, seed=87):
        """Subsample without replacement. Return a new PairedData """
        if n > self.X.shape[0] or n > self.Y.shape[0]:
            raise ValueError('n should not be larger than sizes of X, Y.')
        ind_x = util.subsample_ind( self.X.shape[0], n, seed )
        ind_y = util.subsample_ind( self.Y.shape[0], n, seed )
        return PairedData(self.X[ind_x, :], self.Y[ind_y, :], self.label)

    def clone(self):
        """
        Return a new PairedData object with a separate copy of each internal 
        variable, and with the same content.
        """
        nX = np.copy(self.X)
        nY = np.copy(self.Y)
        nlabel = self.label
        return PairedData(nX, nY, nlabel)

    def __add__(self, pdata2):
        """
        Merge the current PairedData with another one.
        Create a new PairedData and create a new copy for all internal variables.
        label is set to None.
        """
        copy = self.clone()
        copy2 = pdata2.clone()
        nX = np.vstack((copy.X, copy2.X))
        nY = np.vstack((copy.Y, copy2.Y))
        return PairedData(nX, nY)



### end PairedData class        

class PairedSource(object, metaclass=ABCMeta):
    """A data source where it is possible to resample. Subclasses may prefix 
    class names with PS. 

    - If possible, prefix with PSInd to indicate that the 
    PairedSource contains two independent samples. 
    - Prefix with PSDep otherwise.
    - Use PS if the PairedSource can be either one depending on the provided 
    paramters."""

    @abstractmethod
    def sample(self, n, seed):
        """Return a PairedData. Returned result should be deterministic given 
        the input (n, seed)."""
        raise NotImplementedError()

    @abstractmethod
    def dx(self):
        """Return the dimension of X"""
        raise NotImplementedError()
    
    @abstractmethod
    def dy(self):
        """Return the dimension of Y"""
        raise NotImplementedError()
        

class PSResample(PairedSource):
    """
    A PairedSource which subsamples without replacement from the specified
    PairedData.
    """

    def __init__(self, pdata):
        self.pdata = pdata

    def sample(self, n, seed=900):
        pdata_sub = self.pdata.subsample(n, seed)
        return pdata_sub

    def dx(self):
        return self.pdata.dx()

    def dy(self):
        return self.pdata.dy()


class PSStraResample(PairedSource):
    """
    A PairedSource which does a stratified subsampling. without replacement
    from the specified PairedData. 
    The implementation is only approximately correctly.
    """
    def __init__(self, pdata, pivot):
        """
        pivot: a one-dimensional numpy array of the same size as pdata.sample_size()
            indicating the class of each point.
        """
        if len(pivot) != pdata.sample_size():
            raise ValueError('pivot must have the same length as the data.')
        self.pdata = pdata
        self.pivot = pivot
        uniq, counts = np.unique(pivot, return_counts=True)
        self._uniques = uniq
        self._counts = counts

    def sample(self, n, seed=900):
        pdata = self.pdata
        n_sam = pdata.sample_size()
        if n > n_sam:
            raise ValueError('Cannot subsample %d points from %d points.'%(n, n_sam))

        X, Y = pdata.xy()
        import math
        # permute X, Y. Keep pairs 
        I = util.subsample_ind(n_sam, n_sam, seed=seed+3)
        X = X[I, :]
        Y = Y[I, :]
        perm_pivot = self.pivot[I]
        list_chosenI = []
        for ui, v in enumerate(self._uniques):
            Iv  = np.nonzero(np.abs(perm_pivot - v) <= 1e-8)
            Iv = Iv[0]
            niv = self._counts[ui]
            # ceil guarantees that at least 1 instance will be chosen 
            # from each class. 
            n_class = int(math.ceil(niv/float(n_sam)*n))
            chosenI = Iv[:n_class]
            #print chosenI
            list_chosenI.append(chosenI)
        final_chosenI = np.hstack(list_chosenI)
        #print final_chosenI
        reduceI = util.subsample_ind(len(final_chosenI), min(n, len(final_chosenI)), seed+5)
        final_chosenI = final_chosenI[reduceI]
        assert len(final_chosenI) == n, 'final_chosenI has length %d which is not n=%d'%(len(final_chosenI), n)

        Xsam = X[final_chosenI, :]
        Ysam = Y[final_chosenI, :]
        new_label = None if pdata.label is None else pdata.label + '_stra'
        return PairedData(Xsam, Ysam, label=new_label)

    def dx(self):
        return self.pdata.dx()

    def dy(self):
        return self.pdata.dy()

# end PSStraResample

class PSNullShuffle(PairedSource):
    """
    Randomly permute the order of one sample so that the pairs are guaranteed
    to be broken. This is very similar to PSNullResample except it does not 
    sample. The sampling part is delegated to another PairedSource. 
    Essentially, PSNullResample = PSNullShuffle + PSResample.
    Decorator pattern.
    """
    def __init__(self, ps):
        self.ps = ps 

    def sample(self, n, seed=7):
        if n == 1:
            pdata = self.ps.sample(2, seed=seed+27)
            X, Y = pdata.xy()
            nX = X[[0], :]
            nY = Y[[1], :]
        else:
            pdata = self.ps.sample(n, seed=seed+27)
            nX, Y = pdata.xy()
            ind_shift1 = np.roll(list(range(n)), 1)
            nY = Y[ind_shift1, :]

        new_label = 'null_shuffle'
        return PairedData(nX, nY, label=new_label)

    def dx(self):
        return self.ps.dx()

    def dy(self):
        return self.ps.dy()

# end PSNullShuffle



class PSNullResample(PairedSource):
    """
    Randomly permute the order of one sample so that the pairs are guaranteed
    to be broken. This is meant to simulate the case where [H0: X, Y are
    independent] is true.

    A PairedSource which subsamples without replacement from the permuted two
    samples.
    """

    def __init__(self, pdata):
        """
        pdata: A PairedData object
        """
        self.pdata = pdata

    def sample(self, n, seed=981):
        if n > self.pdata.sample_size():
            raise ValueError('cannot sample more points than what the original dataset has')
        X, Y =  self.pdata.xy()
        if n == 1:
            ind = util.subsample_ind(self.pdata.sample_size(), 2, seed=seed)
            nX = X[[ind[0]], :] 
            nY = Y[[ind[1]], :]
        else:
            ind = util.subsample_ind(self.pdata.sample_size(), n, seed=seed)
            nX = X[ind, :]
            ind_shift1 = np.roll(ind, 1)
            nY = Y[ind_shift1, :]
        new_label = None if self.pdata.label is None else self.pdata.label + '_shuf'
        return PairedData(nX, nY, label=new_label)

    def dx(self):
        return self.pdata.dx()

    def dy(self):
        return self.pdata.dy()

# end class PSNullResample

class PSStandardize(PairedSource):
    """
    A PairedSource that standardizes dimensions of X, Y independently so that 
    each has 0 mean and unit variance. Useful with PSResample or PSNullResample 
    when working with real data whose variables do not have the same scaling.

    Decorator pattern.
    """
    def __init__(self, ps):
        """
        ps: a PairedSource
        """
        self.ps = ps 

    def sample(self, n, seed=55):
        ps = self.ps 
        pdata = ps.sample(n, seed=seed)
        X, Y = pdata.xy()

        Zx = util.standardize(X)
        Zy = util.standardize(Y)
        assert np.all(np.isfinite(Zx))
        assert np.all(np.isfinite(Zy))
        new_label = None if pdata.label is None else pdata.label + '_std'
        return PairedData(Zx, Zy, label=new_label)
    
    def dx(self):
        return self.ps.dx()

    def dy(self):
        return self.ps.dy()

# end of class PSStandardize

class PSGaussNoiseDims(PairedSource):
    """
    A PairedSource that adds noise dimensions to X, Y drawn from the specified 
    PairedSource. The noise follows the standard normal distribution.

    Decorator pattern.
    """
    def __init__(self, ps, ndx, ndy):
        """
        ndx: number of noise dimensions for X 
        ndy: number of noise dimensions for Y 
        """
        assert ndx >= 0
        assert ndy >= 0
        self.ps = ps 
        self.ndx = ndx 
        self.ndy = ndy
    
    def sample(self, n, seed=44):
        with util.NumpySeedContext(seed=seed+100):
            NX = np.random.randn(n, self.ndx)
            NY = np.random.randn(n, self.ndy)

            pdata = self.ps.sample(n, seed=seed)
            X, Y = pdata.xy()
            Zx = np.hstack((X, NX))
            Zy = np.hstack((Y, NY))
            new_label = None if pdata.label is None else \
                pdata.label + '_ndx%d'%self.ndx + '_ndy%d'%self.ndy
            return PairedData(Zx, Zy, label=new_label)

    def dx(self):
        return self.ps.dx() + self.ndx

    def dy(self):
        return self.ps.dy() + self.ndy

# end of class PSGaussNoiseDims

class PSFunc(PairedSource):
    """
    A PairedSource that generates data (X, Y) such that Y = f(X) for a 
    specified function f (possibly stochastic), and px where X ~ px.
    """

    def __init__(self, f, px):
        """
        f: function such that Y = f(X). (n x dx)  |-> n x dy
        px: prior on X. Used to generate X. n |-> n x dx
        """
        self.f = f 
        self.px = px
        x = px(2)
        y = f(x)
        self.dx = x.shape[1]
        self.dy = y.shape[1]

    def sample(self, n, seed):
        rstate = np.random.get_state()
        np.random.seed(seed)

        px = self.px 
        X = px(n )
        f = self.f 
        Y = f(X )

        np.random.set_state(rstate)
        return PairedData(X, Y, label='psfunc')

    def dx(self):
        return self.dx

    def dy(self):
        return self.dy

class PSUnifRotateNoise(PairedSource):
    """
    X, Y are dependent in the same way as in PS2DUnifRotate. However, this 
    problem adds more extra noise dimensions.
    - The total number of dimensions is 2+2*noise_dim.
    - Only the first dimensions of X and Y are dependent. Dependency strength 
        depends on the specified angle.
    """

    def __init__(self, angle, xlb=-1, xub=1, ylb=-1, yub=1, noise_dim=0):
        """
        angle: angle in radian
        xlb: lower bound for x (a real number)
        xub: upper bound for x (a real number)
        ylb: lower bound for y (a real number)
        yub: upper bound for y (a real number)
        noise_dim: number of noise dimensions to add to each of X and Y. All
            the extra dimensions follow U(-1, 1)
        """
        ps_2d_unif = PS2DUnifRotate(angle, xlb=xlb, xub=xub, ylb=ylb, yub=yub)
        self.ps_2d_unif = ps_2d_unif
        self.noise_dim = noise_dim

    def sample(self, n, seed=883):
        sample2d = self.ps_2d_unif.sample(n, seed)
        noise_dim = self.noise_dim
        if noise_dim <= 0:
            return sample2d

        rstate = np.random.get_state()
        np.random.seed(seed+1)

        # draw n*noise_dim points from U(-1, 1)
        Xnoise = stats.uniform.rvs(loc=-1, scale=2,
                size=noise_dim*n).reshape(n, noise_dim)
        Ynoise = stats.uniform.rvs(loc=-1, scale=2,
                size=noise_dim*n).reshape(n, noise_dim)

        # concatenate the noise dims to the 2d problem 
        X2d, Y2d = sample2d.xy()
        X = np.hstack((X2d, Xnoise))
        Y = np.hstack((Y2d, Ynoise))

        np.random.set_state(rstate)

        return PairedData(X, Y, label='rot_unif_noisedim%d'%(noise_dim))

    def dx(self):
        return 1+ self.noise_dim

    def dy(self):
        return 1+ self.noise_dim


class PS2DSinFreq(PairedSource):
    """
    X, Y follow the density proportional to 1+sin(w*x)sin(w*y) where 
    w is the frequency. The higher w, the close the density is to a uniform
    distribution on [-pi, pi] x [-pi, pi].

    This dataset was used in Arthur Gretton's lecture notes.
    """
    def __init__(self, freq):
        """
        freq: a nonnegative floating-point number
        """
        self.freq = freq

    def sample(self, n, seed=81):
        ps = PSSinFreq(self.freq, d=1)
        pdata = ps.sample(n, seed=seed)
        X, Y = pdata.xy()

        return PairedData(X, Y, label='sin_freq%.2f'%self.freq)

    def _sample_sequential(self, n, seed=81):
        """
        With a loop, slow.
        """
        rstate = np.random.get_state()
        np.random.seed(seed)

        # rejection sampling
        w = self.freq
        sam = np.zeros((n, 2))
        ind = 0
        #unif_den = 1.0/(4*math.pi**2)
        #ref_bound = 2.0/unif_den
        while ind<n:
            # uniformly randomly draw x, y from U(-pi, pi)
            x = stats.uniform.rvs(loc=-math.pi, scale=2*math.pi, size=1)
            y = stats.uniform.rvs(loc=-math.pi, scale=2*math.pi, size=1)
            if stats.uniform.rvs() < (1+np.sin(w*x)*np.sin(w*y))/2.0:
                # accept 
                sam[ind, :] = [x, y]
                ind = ind + 1

        np.random.set_state(rstate)
        return PairedData(sam[:, [0]], sam[:, [1]], label='sin_freq%.2f'%self.freq)


    def dx(self):
        return 1
    
    def dy(self):
        return 1

# end class PS2DSinFreq

class PSSinFreq(PairedSource):
    """
    X, Y follow the density proportional to 
        1+\prod_{i=1}^{d} [ sin(w*x_i)sin(w*y_i) ]
    w is the frequency. The higher w, the close the density is to a uniform
    distribution on [-pi, pi] x [-pi, pi].
    - This is a generalization of PS2DSinFreq.
    """
    def __init__(self, freq, d):
        """
        freq: a nonnegative floating-point number
        """
        self.freq = freq
        self.d = d

    def sample(self, n, seed=81):
        d = self.d
        Sam = PSSinFreq.sample_d_variates(self.freq, n, 2*self.d, seed)
        X = Sam[:, :d]
        Y = Sam[:, d:]
        return PairedData(X, Y, label='sin_freq%.2f_d%d'%(self.freq, d) )

    def dx(self):
        return self.d

    def dy(self):
        return self.d

    @staticmethod 
    def sample_d_variates(w, n, D, seed=81):
        """
        Return an n x D sample matrix. 
        """
        with util.NumpySeedContext(seed=seed):
            # rejection sampling
            sam = np.zeros((n, D))
            # sample block_size*D at a time.
            block_size = 500
            from_ind = 0
            while from_ind < n:
                # uniformly randomly draw x, y from U(-pi, pi)
                X = stats.uniform.rvs(loc=-math.pi, scale=2*math.pi, size=D*block_size)
                X = np.reshape(X, (block_size, D))
                un_den = 1.0+np.prod(np.sin(w*X), 1)
                I = stats.uniform.rvs(size=block_size) < un_den/2.0

                # accept 
                accepted_count = np.sum(I)
                to_take = min(n - from_ind, accepted_count)
                end_ind = from_ind + to_take

                AX = X[I, :]
                X_take = AX[:to_take, :]
                sam[from_ind:end_ind, :] = X_take
                from_ind = end_ind
        return sam



class PS2DUnifRotate(PairedSource):
    """
    X, Y follow uniform distributions (default to U(-1, 1)). Rotate them by a
    rotation matrix of the specified angle. This can be used to simulate the
    setting of an ICA problem.
    """
    def __init__(self, angle, xlb=-1, xub=1, ylb=-1, yub=1):
        """
        angle: angle in radian
        xlb: lower bound for x (a real number)
        xub: upper bound for x (a real number)
        ylb: lower bound for y (a real number)
        yub: upper bound for y (a real number)
        """
        self.angle = angle
        self.xlb = xlb
        self.xub = xub 
        self.ylb = ylb 
        self.yub = yub

    def sample(self, n, seed=389):
        t = self.angle
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

        ps_unif = PSIndUnif(xlb=[self.xlb], xub=[self.xub], ylb=[self.ylb], yub=[self.yub])
        pdata = ps_unif.sample(n, seed)
        X, Y = pdata.xy()
        XY = np.hstack((X, Y))
        rot_XY = XY.dot(rot.T)

        return PairedData(rot_XY[:, [0]], rot_XY[:, [1]], label='rot_unif_a%.2f'%(t))

    def dx(self):
        return 1

    def dy(self):
        return 1


class PSIndUnif(PairedSource):
    """
    Multivariate (or univariate) uniform distributions for both X, Y
    on the specified boundaries
    """

    def __init__(self, xlb, xub, ylb, yub):
        """
        xlb: a numpy array of lower bounds of x
        xub: a numpy array of upper bounds of x
        ylb: a numpy array of lower bounds of y
        yub: a numpy array of upper bounds of y
        """
        convertif = lambda a: np.array(a) if isinstance(a, list) else a 
        xlb, xub, ylb, yub = list(map(convertif, [xlb, xub, ylb, yub]))
        if xlb.shape[0] != xub.shape[0]:
            raise ValueError('lower and upper bounds of X must be of the same length.')

        if ylb.shape[0] != yub.shape[0]:
            raise ValueError('lower and upper bounds of X must be of the same length.')

        if not np.all(xub - xlb > 0):
            raise ValueError('Require upper - lower to be positive. False for x')

        if not np.all(yub - ylb > 0):
            raise ValueError('Require upper - lower to be positive. False for y')

        self.xlb = xlb
        self.xub = xub 
        self.ylb = ylb 
        self.yub = yub

    def sample(self, n, seed):
        rstate = np.random.get_state()
        np.random.seed(seed)

        dx = self.xlb.shape[0]
        dy = self.ylb.shape[0]
        X = np.zeros((n, dx)) 
        Y = np.zeros((n, dy)) 

        pscale = self.xub - self.xlb 
        qscale = self.yub - self.ylb
        for i in range(dx):
            X[:, i] = stats.uniform.rvs(loc=self.xlb[i], scale=pscale[i], size=n)
        for i in range(dy):
            Y[:, i] = stats.uniform.rvs(loc=self.ylb[i], scale=qscale[i], size=n)

        np.random.set_state(rstate)
        return PairedData(X, Y, label='ind_unif_dx%d_dy%d'%(dx, dy))

    def dx(self):
        return self.xlb.shape[0]

    def dy(self):
        return self.ylb.shape[0]


class PSIndSameGauss(PairedSource):
    """Two same standard Gaussians for P, Q.  """
    def __init__(self, dx, dy):
        """
        dx: dimension of X
        dy: dimension of Y
        """
        self.dimx = dx 
        self.dimy = dy

    def sample(self, n, seed):
        rstate = np.random.get_state()
        np.random.seed(seed)

        X = np.random.randn(n, self.dx())
        Y = np.random.randn(n, self.dy()) 
        np.random.set_state(rstate)
        return PairedData(X, Y, label='sg_dx%d_dy%d'%(self.dx(), self.dy()) )

    def dx(self):
        return self.dimx

    def dy(self):
        return self.dimy

# end class PSIndSameGauss


class PSPairwiseSign(PairedSource):
    """
    A toy problem given in section 5.3 of 

    Large-Scale Kernel Methods for Independence Testing
    Qinyi Zhang, Sarah Filippi,  Arthur Gretton, Dino Sejdinovic

    X ~ N(0, I_d)
    Y = \sqrt(2/d) \sum_{j=1}^{d/2} sign(X_{2j-1 * X_{2j}})|Z_j| + Z_{d/2+1}

    where Z ~ N(0, I_{d/2+1})
    """

    def __init__(self, dx):
        """
        dx: the dimension of X
        """
        if dx <= 0 or dx%2 != 0:
            raise ValueError('dx has to be even')
        self.dimx = dx

    def sample(self, n, seed):
        d = self.dimx 
        with util.NumpySeedContext(seed=seed):
            Z = np.random.randn(n, d/2+1)
            X = np.random.randn(n, d)
            Y = np.zeros((n, 1))
            for j in range(d/2):
                Y = Y + np.sign(X[:, [2*j]]*X[:, [2*j+1]])*np.abs(Z[:, [j]])
            Y = np.sqrt(2.0/d)*Y + Z[:, [d/2]]
        return PairedData(X, Y, label='pairwise_sign_dx%d'%self.dimx)


    def dx(self):
        return self.dimx

    def dy(self):
        return 1

# end class PSPairwiseSign


class PSGaussSign(PairedSource):
    """
    A toy problem where X follows the standard multivariate Gaussian, 
    and Y = sign(product(X))*|Z| where Z ~ N(0, 1). 
    """

    def __init__(self, dx):
        """
        dx: the dimension of X 
        """
        if dx <= 0:
            raise ValueError('dx must be > 0')
        self.dimx = dx 

    def sample(self, n, seed):
        d = self.dimx 
        with util.NumpySeedContext(seed=seed):
            Z = np.random.randn(n, 1)
            X = np.random.randn(n, d)
            Xs = np.sign(X)
            Y = np.prod(Xs, 1)[:, np.newaxis]*np.abs(Z)
        return PairedData(X, Y, label='gauss_sign_dx%d'%d)


    def dx(self):
        return self.dimx 

    def dy(self):
        return 1



