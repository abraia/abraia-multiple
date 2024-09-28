import cv2
import math
import textwrap
import numpy as np

from abc import ABC, abstractmethod


REFERENCE_FACIAL_POINTS = [[38.2946, 51.6963],
                           [73.5318, 51.5014],
                           [56.0252, 71.7366],
                           [41.5493, 92.3655],
                           [70.7299, 92.2041]]


ref_pts = np.array(REFERENCE_FACIAL_POINTS, dtype=np.float32)


def similarity_transform(src_pts, ref_pts):
    tform = SimilarityTransform()
    tform.estimate(src_pts, ref_pts)
    M = tform.params[0:2, :]
    return M


def affine_transform(src_pts, ref_pts):
    # M, _ = cv2.estimateAffine2D(src_pts, ref_pts)
    M, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts)
    return M


def align_face(img, src_pts, size=112):
    dst_pts = ref_pts * size / 112 if size != 112 else ref_pts
    M = similarity_transform(src_pts, dst_pts)
    # M = affine_transform(src_pts, ref_pts)
    warped = cv2.warpAffine(img, M, (size, size), borderValue=0.0)
    return warped


def align_faces(img, results, size=112):
    imgs = []
    for result in results:
        src_pts = result['keypoints']
        imgs.append(align_face(img, src_pts, size))
    return imgs


def _umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array_like
        Source coordinates.
    dst : (M, N) array_like
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`

    """
    src = np.asarray(src)
    dst = np.asarray(dst)

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


class _GeometricTransform(ABC):
    """Abstract base class for geometric transformations."""

    @abstractmethod
    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array_like
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Destination coordinates.

        """

    @property
    @abstractmethod
    def inverse(self):
        """Return a transform object representing the inverse."""

    def residuals(self, src, dst):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the Euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N,) array
            Residual for coordinate.

        """
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))


class ProjectiveTransform(_GeometricTransform):
    r"""Projective transformation.

    Apply a projective transformation (homography) on coordinates.

    For each homogeneous coordinate :math:`\mathbf{x} = [x, y, 1]^T`, its
    target position is calculated by multiplying with the given matrix,
    :math:`H`, to give :math:`H \mathbf{x}`::

      [[a0 a1 a2]
       [b0 b1 b2]
       [c0 c1 1 ]].

    E.g., to rotate by theta degrees clockwise, the matrix should be::

      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]

    or, to translate x by 10 and y by 20::

      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].

    Parameters
    ----------
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix.
    dimensionality : int, optional
        The number of dimensions of the transform. This is ignored if
        ``matrix`` is not None.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix=None, *, dimensionality=2):
        if matrix is None:
            # default to an identity transform
            matrix = np.eye(dimensionality + 1)
        else:
            matrix = np.asarray(matrix)
            dimensionality = matrix.shape[0] - 1
            if matrix.shape != (dimensionality + 1, dimensionality + 1):
                raise ValueError("invalid shape of transformation matrix")
        self.params = matrix
        self._coeffs = range(matrix.size - 1)

    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)

    def _apply_mat(self, coords, matrix):
        ndim = matrix.shape[0] - 1
        coords = np.array(coords, copy=NP_COPY_IF_NEEDED, ndmin=2)

        src = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
        dst = src @ matrix.T

        # below, we will divide by the last dimension of the homogeneous
        # coordinate matrix. In order to avoid division by zero,
        # we replace exact zeros in this column with a very small number.
        dst[dst[:, ndim] == 0, ndim] = np.finfo(float).eps
        # rescale to homogeneous coordinates
        dst[:, :ndim] /= dst[:, ndim : ndim + 1]

        return dst[:, :ndim]

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            return self.params
        else:
            return self.params.astype(dtype)

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, D) array_like
            Source coordinates.

        Returns
        -------
        coords_out : (N, D) array
            Destination coordinates.

        """
        return self._apply_mat(coords, self.params)

    @property
    def inverse(self):
        """Return a transform object representing the inverse."""
        return type(self)(matrix=self._inv_matrix)

    def estimate(self, src, dst, weights=None):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = (a0*x + a1*y + a2) / (c0*x + c1*y + 1)
            Y = (b0*x + b1*y + b2) / (c0*x + c1*y + 1)

        These equations can be transformed to the following form::

            0 = a0*x + a1*y + a2 - c0*x*X - c1*y*X - X
            0 = b0*x + b1*y + b2 - c0*x*Y - c1*y*Y - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[x y 1 0 0 0 -x*X -y*X -X]
                   [0 0 0 x y 1 -x*Y -y*Y -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c0 c1 c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalised, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

        In case of the affine transformation the coefficients c0 and c1 are 0.
        Thus the system of equations is::

            A   = [[x y 1 0 0 0 -X]
                   [0 0 0 x y 1 -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c3]

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.
        weights : (N,) array_like, optional
            Relative weight values for each pair of points.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        src = np.asarray(src)
        dst = np.asarray(dst)
        n, d = src.shape

        src_matrix, src = _center_and_normalize_points(src)
        dst_matrix, dst = _center_and_normalize_points(dst)
        if not np.all(np.isfinite(src_matrix + dst_matrix)):
            self.params = np.full((d + 1, d + 1), np.nan)
            return False

        # params: a0, a1, a2, b0, b1, b2, c0, c1
        A = np.zeros((n * d, (d + 1) ** 2))
        # fill the A matrix with the appropriate block matrices; see docstring
        # for 2D example — this can be generalised to more blocks in the 3D and
        # higher-dimensional cases.
        for ddim in range(d):
            A[ddim * n : (ddim + 1) * n, ddim * (d + 1) : ddim * (d + 1) + d] = src
            A[ddim * n : (ddim + 1) * n, ddim * (d + 1) + d] = 1
            A[ddim * n : (ddim + 1) * n, -d - 1 : -1] = src
            A[ddim * n : (ddim + 1) * n, -1] = -1
            A[ddim * n : (ddim + 1) * n, -d - 1 :] *= -dst[:, ddim : (ddim + 1)]

        # Select relevant columns, depending on params
        A = A[:, list(self._coeffs) + [-1]]

        # Get the vectors that correspond to singular values, also applying
        # the weighting if provided
        if weights is None:
            _, _, V = np.linalg.svd(A)
        else:
            weights = np.asarray(weights)
            W = np.diag(np.tile(np.sqrt(weights / np.max(weights)), d))
            _, _, V = np.linalg.svd(W @ A)

        # if the last element of the vector corresponding to the smallest
        # singular value is close to zero, this implies a degenerate case
        # because it is a rank-defective transform, which would map points
        # to a line rather than a plane.
        if np.isclose(V[-1, -1], 0):
            self.params = np.full((d + 1, d + 1), np.nan)
            return False

        H = np.zeros((d + 1, d + 1))
        # solution is right singular vector that corresponds to smallest
        # singular value
        H.flat[list(self._coeffs) + [-1]] = -V[-1, :-1] / V[-1, -1]
        H[d, d] = 1

        # De-center and de-normalize
        H = np.linalg.inv(dst_matrix) @ H @ src_matrix

        # Small errors can creep in if points are not exact, causing the last
        # element of H to deviate from unity. Correct for that here.
        H /= H[-1, -1]

        self.params = H

        return True

    def __add__(self, other):
        """Combine this transformation with another."""
        if isinstance(other, ProjectiveTransform):
            # combination of the same types result in a transformation of this
            # type again, otherwise use general projective transformation
            if type(self) == type(other):
                tform = self.__class__
            else:
                tform = ProjectiveTransform
            return tform(other.params @ self.params)
        else:
            raise TypeError("Cannot combine transformations of differing " "types.")

    def __nice__(self):
        """common 'paramstr' used by __str__ and __repr__"""
        npstring = np.array2string(self.params, separator=', ')
        paramstr = 'matrix=\n' + textwrap.indent(npstring, '    ')
        return paramstr

    def __repr__(self):
        """Add standard repr formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f'<{classstr}({paramstr}) at {hex(id(self))}>'

    def __str__(self):
        """Add standard str formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f'<{classstr}({paramstr})>'

    @property
    def dimensionality(self):
        """The dimensionality of the transformation."""
        return self.params.shape[0] - 1
    

class EuclideanTransform(ProjectiveTransform):
    """Euclidean transformation, also known as a rigid transform.

    Has the following form::

        X = a0 * x - b0 * y + a1 =
          = x * cos(rotation) - y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = x * sin(rotation) + y * cos(rotation) + b1

    where the homogeneous transformation matrix is::

        [[a0 -b0  a1]
         [b0  a0  b1]
         [0   0   1 ]]

    The Euclidean transformation is a rigid transformation with rotation and
    translation parameters. The similarity transformation extends the Euclidean
    transformation with a single scaling factor.

    In 2D and 3D, the transformation parameters may be provided either via
    `matrix`, the homogeneous transformation matrix, above, or via the
    implicit parameters `rotation` and/or `translation` (where `a1` is the
    translation along `x`, `b1` along `y`, etc.). Beyond 3D, if the
    transformation is only a translation, you may use the implicit parameter
    `translation`; otherwise, you must use `matrix`.

    Parameters
    ----------
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix.
    rotation : float or sequence of float, optional
        Rotation angle, clockwise, as radians. If given as
        a vector, it is interpreted as Euler rotation angles [1]_. Only 2D
        (single rotation) and 3D (Euler rotations) values are supported. For
        higher dimensions, you must provide or estimate the transformation
        matrix.
    translation : (x, y[, z, ...]) sequence of float, length D, optional
        Translation parameters for each axis.
    dimensionality : int, optional
        The dimensionality of the transform.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """

    def __init__(
        self, matrix=None, rotation=None, translation=None, *, dimensionality=2
    ):
        params_given = rotation is not None or translation is not None

        if params_given and matrix is not None:
            raise ValueError(
                "You cannot specify the transformation matrix and"
                " the implicit parameters at the same time."
            )
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
        elif params_given:
            if rotation is None:
                dimensionality = len(translation)
                if dimensionality == 2:
                    rotation = 0
                elif dimensionality == 3:
                    rotation = np.zeros(3)
                else:
                    raise ValueError(
                        'Parameters cannot be specified for dimension '
                        f'{dimensionality} transforms'
                    )
            else:
                if not np.isscalar(rotation) and len(rotation) != 3:
                    raise ValueError(
                        'Parameters cannot be specified for dimension '
                        f'{dimensionality} transforms'
                    )
            if translation is None:
                translation = (0,) * dimensionality

            if dimensionality == 2:
                self.params = np.array(
                    [
                        [math.cos(rotation), -math.sin(rotation), 0],
                        [math.sin(rotation), math.cos(rotation), 0],
                        [0, 0, 1],
                    ]
                )
            elif dimensionality == 3:
                self.params = np.eye(dimensionality + 1)
                self.params[:dimensionality, :dimensionality] = _euler_rotation_matrix(
                    rotation
                )
            self.params[0:dimensionality, dimensionality] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        self.params = _umeyama(src, dst, False)

        # _umeyama will return nan if the problem is not well-conditioned.
        return not np.any(np.isnan(self.params))

    @property
    def rotation(self):
        if self.dimensionality == 2:
            return math.atan2(self.params[1, 0], self.params[1, 1])
        elif self.dimensionality == 3:
            # Returning 3D Euler rotation matrix
            return self.params[:3, :3]
        else:
            raise NotImplementedError(
                'Rotation only implemented for 2D and 3D transforms.'
            )

    @property
    def translation(self):
        return self.params[0 : self.dimensionality, self.dimensionality]


class SimilarityTransform(EuclideanTransform):
    """Similarity transformation.

    Has the following form in 2D::

        X = a0 * x - b0 * y + a1 =
          = s * x * cos(rotation) - s * y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = s * x * sin(rotation) + s * y * cos(rotation) + b1

    where ``s`` is a scale factor and the homogeneous transformation matrix is::

        [[a0 -b0  a1]
         [b0  a0  b1]
         [0   0   1 ]]

    The similarity transformation extends the Euclidean transformation with a
    single scaling factor in addition to the rotation and translation
    parameters.

    Parameters
    ----------
    matrix : (dim+1, dim+1) array_like, optional
        Homogeneous transformation matrix.
    scale : float, optional
        Scale factor. Implemented only for 2D and 3D.
    rotation : float, optional
        Rotation angle, clockwise, as radians.
        Implemented only for 2D and 3D. For 3D, this is given in ZYX Euler
        angles.
    translation : (dim,) array_like, optional
        x, y[, z] translation parameters. Implemented only for 2D and 3D.

    Attributes
    ----------
    params : (dim+1, dim+1) array
        Homogeneous transformation matrix.

    """

    def __init__(
        self,
        matrix=None,
        scale=None,
        rotation=None,
        translation=None,
        *,
        dimensionality=2,
    ):
        self.params = None
        params = any(param is not None for param in (scale, rotation, translation))

        if params and matrix is not None:
            raise ValueError(
                "You cannot specify the transformation matrix and"
                " the implicit parameters at the same time."
            )
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                self.params = matrix
                dimensionality = matrix.shape[0] - 1
        if params:
            if dimensionality not in (2, 3):
                raise ValueError('Parameters only supported for 2D and 3D.')
            matrix = np.eye(dimensionality + 1, dtype=float)
            if scale is None:
                scale = 1
            if rotation is None:
                rotation = 0 if dimensionality == 2 else (0, 0, 0)
            if translation is None:
                translation = (0,) * dimensionality
            if dimensionality == 2:
                ax = (0, 1)
                c, s = np.cos(rotation), np.sin(rotation)
                matrix[ax, ax] = c
                matrix[ax, ax[::-1]] = -s, s
            else:  # 3D rotation
                matrix[:3, :3] = _euler_rotation_matrix(rotation)

            matrix[:dimensionality, :dimensionality] *= scale
            matrix[:dimensionality, dimensionality] = translation
            self.params = matrix
        elif self.params is None:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        self.params = _umeyama(src, dst, estimate_scale=True)

        # _umeyama will return nan if the problem is not well-conditioned.
        return not np.any(np.isnan(self.params))

    @property
    def scale(self):
        # det = scale**(# of dimensions), therefore scale = det**(1/ndim)
        if self.dimensionality == 2:
            return np.sqrt(np.linalg.det(self.params))
        elif self.dimensionality == 3:
            return np.cbrt(np.linalg.det(self.params))
        else:
            raise NotImplementedError('Scale is only implemented for 2D and 3D.')
        