import numpy as np

def DLT(m, M):
    """
    
    Uses Direct Linear Transform (DLT) to get the homography matrix matching m
    with M
    
    Parameters
    ----------
    m : TYPE ndarray
        DESCRIPTION. Image points
    M : TYPE ndarray
        DESCRIPTION. World points
        
    Returns
    -------
    H : TYPE ndarray
        DESCRIPTION. Homography matrix
    """
    
    

    assert m.shape == M.shape

    nb_points = m.shape[0]

    A = np.zeros((nb_points*2, 9))

    for i in range(nb_points):
        X, Y, Z = M[i, 0], M[i, 1], 1
        u, v = m[i, 0], m[i, 1]

        A[2*i, :] = [-X, -Y, -Z, 0, 0, 0, u*X, u*Y, u*Z]
        A[2*i + 1, :] = [0, 0, 0, -X, -Y, -Z, v*X, v*Y, v*Z]

    # Calculate homography matrix H using SVD
    U, S, V = np.linalg.svd(A)
    H = V[-1, :].reshape(3, 3)

    # Normalize H such that H[2, 2] = 1
    H = H / H[2, 2]

    return H


def RT_from_3D(m, M, K):
    """
    
    Finds the rotation and translation matrix of camera given image plane
    points m, world points M, and camera intrinsics. Note that this algorithm
    works because M lies on a plane (it doesn't have a Z component)
    
    Parameters
    ----------
    m : TYPE ndarray
        DESCRIPTION. Image points
    M : TYPE ndarray
        DESCRIPTION. World points
    K : TYPE ndarray
        DESCRIPTION. Camera intrinsics

    Returns
    -------
    R : TYPE ndarray
        DESCRIPTION. Rotation matrix
    T : TYPE ndarray
        DESCRIPTION. Translation vector

    """
    
    nb_points = M.shape[0]

    # Convert M and m to homogeneous coordinates
    M_homog = np.hstack((M, np.ones((nb_points, 1))))
    m_homog = np.hstack((m, np.ones((nb_points, 1))))

    # Normalize image coordinates
    m_norm = np.linalg.inv(K) @ m_homog.T
    m_norm = m_norm[:2, :].T

    # Compute the homography
    H = DLT(m_homog, M_homog)

    # Compute the rotation and translation
    H_norm = np.linalg.inv(K) @ H
    norm_factor = np.sqrt(np.sum(np.power(H_norm[:, 0], 2)))

    r1 = H_norm[:, 0] / norm_factor
    r2 = H_norm[:, 1] / norm_factor
    t = H_norm[:, 2] / norm_factor
    
    R = np.column_stack((r1, r2, np.cross(r1, r2)))
    T = t.reshape(3, 1)

    return R, T


    
    