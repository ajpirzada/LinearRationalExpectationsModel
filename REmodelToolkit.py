# =============================================================================
# By Ahmed Pirzada, University of Bristol
# aj.pirzada@bristol.ac.uk
# 5th March 2025
# =============================================================================


# =============================================================================
# Import Python libraries
# =============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
import scipy.linalg as la
from tabulate import tabulate


# =============================================================================
# Python adaptation of Chris Sim's Gensys algorithm
# =============================================================================
def gensys(g0, g1, c, psi, pi, div=None):
    """
    Solves the linear rational expectations model

           g0*y(t) = g1*y(t-1) + c + psi*z(t) + pi*eta(t),

    where z is an exogenous variable process and eta represents one‐step-ahead expectational errors.
    The resulting system is expressed as

           y(t) = G1*y(t-1) + C + impact*z(t) + ywt*inv(I-fmat*inv(L))*fwt*z(t+1).

    If z(t) is i.i.d., the last term drops out.

    Parameters
    ----------
    g0 : ndarray
         Matrix g0.
    g1 : ndarray
         Matrix g1.
    c : ndarray
         Constant term.
    psi : ndarray
         Coefficient matrix on exogenous shocks.
    pi : ndarray
         Coefficient matrix on expectational errors.
    div : float, optional
         If provided, used as the threshold divisor; otherwise, a value greater than 1 is calculated.

    Returns
    -------
    G1 : ndarray
         Transition matrix for y(t-1).
    C : ndarray
         Constant term.
    impact : ndarray
         Impact matrix on z(t).
    fmat : ndarray
         f-matrix.
    fwt : ndarray
         f-weight matrix.
    ywt : ndarray
         y-weight matrix.
    gev : ndarray
         Array of generalised eigenvalues (each row is [eig_A, eig_B]).
    eu : ndarray
         2-element array: eu[0]=1 for existence, eu[1]=1 for uniqueness.
    loose : ndarray
         Loose endogenised errors.
    """
    realsmall = 1e-6
    eu = np.zeros(2, dtype=int)
    fixdiv = (div is not None)
    n = g0.shape[0]

    # QZ decomposition. In Python, la.qz returns (A, B, Q, Z) so that Q.T*g0*Z = A and Q.T*g1*Z = B.
    a, b, q, z = la.qz(g0, g1, output='real')
    q = q.T             # To be consistent with matlab convention
    if not fixdiv:
        div = 1.01

    nunstab = 0
    zxz = False
    for i in range(n):
        if abs(a[i, i]) > 0:
            divhat = abs(b[i, i]) / abs(a[i, i])
            if (1 + realsmall) < divhat <= div:
                div = 0.5 * (1 + divhat)
        if abs(b[i, i]) > div * abs(a[i, i]):
            nunstab += 1
        if abs(a[i, i]) < realsmall and abs(b[i, i]) < realsmall:
            zxz = True

    if not zxz:
        a, b, q, z = qzdiv(div, a, b, q, z)
    else:
        print('Coincident zeros.  Indeterminacy and/or nonexistence.')
        eu[:] = -2
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), eu, np.array([]))

    gev = np.column_stack((np.diag(a), np.diag(b)))

    # Partition matrices as in Sims’ code.
    q1 = q[:n - nunstab, :]
    q2 = q[n - nunstab:, :]
    # (z1 and z2 are computed in the Matlab code but not used further.)
    z1 = z[:, :n - nunstab].T
    z2 = z[:, n - nunstab:].T

    a2 = a[n - nunstab:n, n - nunstab:n]
    b2 = b[n - nunstab:n, n - nunstab:n]

    etawt = q2 @ pi
    neta = pi.shape[1]

    if nunstab == 0:
        ueta = np.empty((0, 0))
        deta = np.empty((0, 0))
        veta = np.empty((neta, 0))
        bigev = np.array([], dtype=int)
    else:
        U, s, Vh = la.svd(etawt, full_matrices=False)
        md = min(U.shape[0], Vh.shape[0])
        bigev = np.where(s[:md] > realsmall)[0]
        if bigev.size:
            ueta = U[:, bigev]
            deta = np.diag(s[bigev])
            veta = Vh.conj().T[:, bigev]
        else:
            ueta = np.empty((etawt.shape[0], 0))
            deta = np.empty((0, 0))
            veta = np.empty((neta, 0))
    eu[0] = 1 if (bigev.size >= nunstab) else 0

    if nunstab == n:
        etawt1 = np.empty((0, neta))
        ueta1 = np.empty((0, 0))
        deta1 = np.empty((0, 0))
        veta1 = np.empty((neta, 0))
    else:
        etawt1 = q1 @ pi
        ndeta1 = min(n - nunstab, neta)
        U1, s1, Vh1 = la.svd(etawt1, full_matrices=False)
        md1 = min(U1.shape[0], Vh1.shape[0])
        bigev1 = np.where(s1[:md1] > realsmall)[0]
        if bigev1.size:
            ueta1 = U1[:, bigev1]
            deta1 = np.diag(s1[bigev1])
            veta1 = Vh1.conj().T[:, bigev1]
        else:
            ueta1 = np.empty((etawt1.shape[0], 0))
            deta1 = np.empty((0, 0))
            veta1 = np.empty((neta, 0))

    if veta1.size == 0:
        unique = True
    else:
        loose_temp = veta1 - veta @ (veta.T @ veta1)
        _, s_loose, _ = la.svd(loose_temp, full_matrices=False)
        nloose = np.sum(s_loose > realsmall * n)
        unique = (nloose == 0)
    if unique:
        eu[1] = 1
    else:
        print(f'Indeterminacy.  {nloose} loose endog errors.')

    # Construct transformation matrix, tmat.
    if ueta.size and deta.size and veta.size and veta1.size and deta1.size and ueta1.size:
        inner_term = la.solve(deta, veta.T)
        T_term = ueta @ inner_term @ (veta1 @ (deta1 @ ueta1.T))
    else:
        T_term = np.zeros((n - nunstab, 0))
    tmat = np.hstack([np.eye(n - nunstab), -T_term.T])

    # Build G0 and G1.
    upper_G0 = tmat @ a
    lower_G0 = np.hstack([np.zeros((nunstab, n - nunstab)), np.eye(nunstab)])
    G0 = np.vstack([upper_G0, lower_G0])

    upper_G1 = tmat @ b
    lower_G1 = np.zeros((nunstab, n))
    G1_mat = np.vstack([upper_G1, lower_G1])

    G0I = la.inv(G0)
    G1 = G0I @ G1_mat

    usix = np.arange(n - nunstab, n)

    top_C = tmat @ (q @ c)
    A_diff = a[np.ix_(usix, usix)] - b[np.ix_(usix, usix)]
    bottom_C = la.solve(A_diff, q2 @ c)
    C_vec = np.vstack([top_C, bottom_C])
    C = G0I @ C_vec

    top_impact = tmat @ (q @ psi)
    bottom_impact = np.zeros((nunstab, psi.shape[1]))
    impact_vec = np.vstack([top_impact, bottom_impact])
    impact = G0I @ impact_vec

    fmat = la.solve(b[np.ix_(usix, usix)], a[np.ix_(usix, usix)])
    fwt = -la.solve(b[np.ix_(usix, usix)], q2 @ psi)
    ywt = G0I[:, usix]

    if veta.size:
        loose_mid = etawt1 @ (np.eye(neta) - veta @ veta.T)
    else:
        loose_mid = np.empty((etawt1.shape[0], neta))
    loose_stack = np.vstack([loose_mid, np.zeros((nunstab, neta))])
    loose = G0I @ loose_stack

    # Final transformation as in Sims’ code.
    G1 = np.real(z @ G1 @ z.T)
    C = np.real(z @ C)
    impact = np.real(z @ impact)
    loose = np.real(z @ loose)
    ywt = z @ ywt

    return G1, C, impact, fmat, fwt, ywt, gev, eu, loose


# =============================================================================
# Part of Chris Sim's Gensys toolkit
# =============================================================================
def qzdiv(stake, A, B, Q, Z, v=None):
    """
    Rearranges the QZ matrices so that all cases of

           abs(B(i,i)/A(i,i)) > stake

    are moved to the lower right corner while preserving the orthonormal properties.
    This routine mimics the structure of Sims’s MATLAB qzdiv.

    Parameters
    ----------
    stake : float
            Threshold value.
    A : ndarray
            Quasi-triangular matrix A.
    B : ndarray
            Quasi-triangular matrix B.
    Q : ndarray
            Orthonormal matrix Q.
    Z : ndarray
            Orthonormal matrix Z.
    v : ndarray, optional
            Additional matrix to be rearranged (default is None).

    Returns
    -------
    A, B, Q, Z : ndarrays
            Rearranged matrices.
    """
    vin = (v is not None)
    n, _ = A.shape
    diag_A = np.diag(A)
    diag_B = np.diag(B)
    root = np.column_stack((diag_A.copy(), diag_B.copy()))
    mask = (np.abs(root[:, 0]) < 1.e-13)
    root[mask, 0] = root[mask, 0] - (root[mask, 0] + root[mask, 1])
    root[:, 0] = np.where(np.abs(root[:, 0]) < 1.e-13, 1.e-13, root[:, 0])
    root[:, 1] = root[:, 1] / root[:, 0]

    for i in range(n - 1, -1, -1):
        m = -1
        for j in range(i, -1, -1):
            if (root[j, 1] > stake) or (root[j, 1] < -0.1):
                m = j
                break
        if m == -1:
            return A, B, Q, Z
        for k in range(m, i):
            A, B, Q, Z = qzswitch(k, A, B, Q, Z)
            temp = root[k, 1]
            root[k, 1] = root[k+1, 1]
            root[k+1, 1] = temp
            if vin:
                v[:, [k, k+1]] = v[:, [k+1, k]]
    return A, B, Q, Z


# =============================================================================
# Part of Chris Sim's Gensys toolkit
# =============================================================================
def qzswitch(i, A, B, Q, Z):
    """
    Takes upper triangular matrices A, B and orthonormal matrices Q, Z and interchanges
    the diagonal elements i and i+1 of both A and B while maintaining Q'AZ' and Q'BZ' unchanged.
    
    If the diagonal elements of A and B are zero at matching positions, the returned A will have
    zeros at both diagonal positions. This behaviour is natural when this routine is used to drive all
    zeros on the diagonal of A to the lower right.
    
    Parameters
    ----------
    i : int
        The index at which to perform the switch (0-indexed). The function switches rows and columns i and i+1.
    A : ndarray
        Upper triangular matrix A.
    B : ndarray
        Upper triangular matrix B.
    Q : ndarray
        Orthonormal matrix Q.
    Z : ndarray
        Orthonormal matrix Z.
        
    Returns
    -------
    A, B, Q, Z : ndarrays
        The updated matrices after interchanging the diagonal elements.
    """
    realsmall = np.sqrt(np.finfo(float).eps) * 10

    a_val = A[i, i]
    d_val = B[i, i]
    b_val = A[i, i+1]
    e_val = B[i, i+1]
    c_val = A[i+1, i+1]
    f_val = B[i+1, i+1]

    # If left/right (l.r.) diagonal elements of A are coincident zeros.
    if (abs(c_val) < realsmall) and (abs(f_val) < realsmall):
        if abs(a_val) < realsmall:
            # l.r. coincident zeros with upper left of A equal to zero; do nothing.
            return A, B, Q, Z
        else:
            # l.r. coincident zeros; put 0 in upper left element.
            wz = np.array([b_val, -a_val])
            wz = wz / np.sqrt(np.dot(wz, wz))
            # Construct a 2x2 matrix: first row is wz, second row is [-wz[1], wz[0]]
            wz = np.vstack([wz, np.array([-wz[1], wz[0]])])
            xy = np.eye(2)
    elif (abs(a_val) < realsmall) and (abs(d_val) < realsmall):
        if abs(c_val) < realsmall:
            # Upper left coincident zeros with lower right of A equal to zero; do nothing.
            return A, B, Q, Z
        else:
            # Upper left coincident zeros; put 0 in lower right element of A.
            wz = np.eye(2)
            # Build a row vector from [c  -b]
            xy_row = np.array([c_val, -b_val])
            norm_xy = np.sqrt(np.dot(xy_row, xy_row))
            xy_row = xy_row / norm_xy
            # Form a 2x2 matrix with first row xy_row and second row [ -xy_row[1], xy_row[0] ]
            xy = np.vstack([xy_row, np.array([-xy_row[1], xy_row[0]])])
    else:
        # Usual case.
        # Compute row vector: [c*e - f*b, c*d - f*a]
        wz = np.array([c_val * e_val - f_val * b_val,
                       c_val * d_val - f_val * a_val])
        # Compute row vector: [b*d - e*a, c*d - f*a]
        xy = np.array([b_val * d_val - e_val * a_val,
                       c_val * d_val - f_val * a_val])
        n_val = np.sqrt(np.dot(wz, wz))
        m_val = np.sqrt(np.dot(xy, xy))
        if m_val < np.finfo(float).eps * 100:
            # All elements of A and B are proportional; do nothing.
            return A, B, Q, Z
        # Solve the scalings.
        wz = wz / n_val
        xy = xy / m_val
        # Form the 2x2 matrices:
        # For wz, stack the row vector and its rotated version.
        wz = np.vstack([wz, np.array([-wz[1], wz[0]])])
        # Similarly for xy.
        xy = np.vstack([xy, np.array([-xy[1], xy[0]])])
    
    # Apply the transformations.
    # Update rows i and i+1.
    A[i:i+2, :] = xy @ A[i:i+2, :]
    B[i:i+2, :] = xy @ B[i:i+2, :]
    # Update columns i and i+1.
    A[:, i:i+2] = A[:, i:i+2] @ wz
    B[:, i:i+2] = B[:, i:i+2] @ wz
    # Update the orthonormal matrices.
    Z[:, i:i+2] = Z[:, i:i+2] @ wz
    Q[i:i+2, :] = xy @ Q[i:i+2, :]

    return A, B, Q, Z


# =============================================================================
# Solve the model using Gensys
# =============================================================================
def solve_model(params, g0, g1, c, psi, Pi, shock_keys):
    """
    Solve the model using Gensys and automatically construct the covariance matrix Q
    using the shock_keys defined in the global scope.
    
    Parameters
    ----------
    params : dict
        Dictionary of model parameters.
    g0, g1, c, psi, Pi : np.ndarray
        Matrices from the model setup.
    
    Returns
    -------
    A : np.ndarray
        State transition matrix.
    B : np.ndarray
        Impact matrix mapping shocks to the state.
    C : np.ndarray
        Constant term.
    Q : np.ndarray
        Covariance matrix for the structural shocks.
    eu : (depends on gensys output)
        Additional output from gensys.
    """
    # Solve the model using Gensys:
    G1_sol, C_vec, Impact, _, _, _, _, eu, _ = gensys(g0, g1, c, psi, Pi)
    
    # Shock covariance matrix:
    Q = np.diag([params[k]**2 for k in shock_keys])
    
    # Set up the standard state-space representation:
    A = G1_sol      # State transition matrix
    B = Impact      # Shock impact matrix
    C = C_vec       # Constant term
    
    return A, B, C, Q, eu


# =============================================================================
# Define the Kalman filter
# =============================================================================
def kalman_filter(y, A, B, C, H, Q, R, d=None):
    """
    Standard Kalman Filter:
      - State equation:    x_t = A x_{t-1} + B eps_t + C,   eps_t ~ N(0,Q)
      - Measurement eq.:   y_t = H x_t + v_t,                v_t ~ N(0,R)
    Returns the negative log likelihood.
    """
    T = y.shape[0]
    n_state = A.shape[0]
    n_obs = H.shape[0]

    # If no intercept is provided, default to zero vector:
    if d is None:
        d = np.zeros((n_obs, 1))
    else:
        # Ensure d is a column vector:
        d = d.reshape(-1, 1)

    # Initialize state mean (assume zero) and covariance.
    x = np.zeros((n_state, 1))
    # Try to compute the unconditional covariance of the state via the discrete Lyapunov eq.
    try:
        P = solve_discrete_lyapunov(A, B @ Q @ B.T)
    except Exception as e:
        P = 1e6 * np.eye(n_state)
    
    loglik = 0.0

    for t in range(T):
        # ----- Prediction -----
        x_pred = A @ x + C  # predicted state mean
        P_pred = A @ P @ A.T + B @ Q @ B.T  # predicted covariance

        # ----- Measurement prediction -----
        y_pred = H @ x_pred + d # predicted measurement mean
        S = H @ P_pred @ H.T + R  # innovation covariance

        # Check for positive definiteness:
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return 1e10  # return a large number to penalize non-PD covariance

        # Innovation
        innov = y[t, :].reshape(-1, 1) - y_pred

        # Kalman Gain
        K = P_pred @ H.T @ np.linalg.inv(S)

        # ----- Update -----
        x = x_pred + K @ innov
        P = P_pred - K @ H @ P_pred

        # Contribution to the log likelihood (using the multivariate normal density)
        ll_inc = -0.5 * (n_obs * np.log(2*np.pi) + logdet + (innov.T @ np.linalg.inv(S) @ innov).item())
        loglik += ll_inc

    # Return negative log likelihood (because we will minimize)
    return -loglik




def kalman_filter_store(y, A, B, C, H, Q, R, d=None):
    """
    Run the Kalman filter and store filtered state estimates and predictions.

    Returns:
      x_filt: filtered state estimates (n_state x T)
      P_filt: filtered state covariances (n_state x n_state x T)
      x_pred_list: predicted state estimates (n_state x T)
      P_pred_list: predicted state covariances (n_state x n_state x T)
      loglik: log likelihood (for completeness)
    """
    T = y.shape[0]
    n_state = A.shape[0]
    n_obs = H.shape[0]

    # If no intercept is provided, default to zero vector:
    if d is None:
        d = np.zeros((n_obs, 1))
    else:
        # Ensure d is a column vector:
        d = d.reshape(-1, 1)

    # Storage arrays:
    x_filt = np.zeros((n_state, T))
    P_filt = np.zeros((n_state, n_state, T))
    x_pred_list = np.zeros((n_state, T))
    P_pred_list = np.zeros((n_state, n_state, T))
    
    # Initial state
    x = np.zeros((n_state, 1))
    try:
        P = solve_discrete_lyapunov(A, B @ Q @ B.T)
    except Exception as e:
        P = 1e6 * np.eye(n_state)
    
    loglik = 0.0

    for t in range(T):
        # ----- Prediction -----
        x_pred = A @ x + C  # predicted state mean
        P_pred = A @ P @ A.T + B @ Q @ B.T  # predicted covariance
        x_pred_list[:, t] = x_pred.flatten()
        P_pred_list[:, :, t] = P_pred
        
        # ----- Measurement prediction -----
        y_pred = H @ x_pred + d
        S = H @ P_pred @ H.T + R  # innovation covariance
        
        # Ensure S is PD:
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            S += 1e-6 * np.eye(S.shape[0])
        
        # Innovation
        innov = y[t, :].reshape(-1, 1) - y_pred
        
        # Kalman Gain
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # ----- Update -----
        x = x_pred + K @ innov
        P = P_pred - K @ H @ P_pred
        
        # Save the filtered estimates
        x_filt[:, t] = x.flatten()
        P_filt[:, :, t] = P
        
        # Update log-likelihood (optional)
        ll_inc = -0.5 * (n_obs * np.log(2 * np.pi) + logdet +
                         (innov.T @ np.linalg.inv(S) @ innov).item())
        loglik += ll_inc

    return x_filt, P_filt, x_pred_list, P_pred_list, loglik



def rts_smoother(A, x_filt, P_filt, x_pred_list, P_pred_list, epsilon=1e-6):
    """
    Rauch-Tung-Striebel smoother with regularization and fallback to pseudoinverse.
    
    Parameters
    ----------
    A : np.ndarray
        The state transition matrix.
    x_filt : np.ndarray
        Filtered state estimates (n_state x T).
    P_filt : np.ndarray
        Filtered state covariance matrices (n_state x n_state x T).
    x_pred_list : np.ndarray
        Predicted state estimates (n_state x T).
    P_pred_list : np.ndarray
        Predicted state covariance matrices (n_state x n_state x T).
    epsilon : float, optional
        Regularization constant (default is 1e-6).
    
    Returns
    -------
    x_smooth : np.ndarray
        Smoothed state estimates (n_state x T).
    P_smooth : np.ndarray
        Smoothed state covariance matrices (n_state x n_state x T).
    """
    T = x_filt.shape[1]
    n_state = A.shape[0]
    
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    
    # Initialize with the last filtered estimates
    x_smooth[:, -1] = x_filt[:, -1]
    P_smooth[:, :, -1] = P_filt[:, :, -1]
    
    # Backward recursion
    for t in range(T - 2, -1, -1):
        P_filt_t = P_filt[:, :, t]
        P_pred_next = P_pred_list[:, :, t + 1]
        # Regularize P_pred_next:
        P_pred_next_reg = P_pred_next + epsilon * np.eye(P_pred_next.shape[0])
        # Attempt to invert using np.linalg.inv; if that fails, use np.linalg.pinv.
        try:
            inv_term = np.linalg.inv(P_pred_next_reg)
        except np.linalg.LinAlgError:
            inv_term = np.linalg.pinv(P_pred_next_reg)
        # Compute smoother gain:
        J = P_filt_t @ A.T @ inv_term
        # Update state estimate:
        x_smooth[:, t] = x_filt[:, t] + J @ (x_smooth[:, t + 1] - x_pred_list[:, t + 1])
        # Update covariance:
        P_smooth[:, :, t] = P_filt_t + J @ (P_smooth[:, :, t + 1] - P_pred_next_reg) @ J.T
        
    return x_smooth, P_smooth




def recover_shocks(A, B, C, x_smooth):
    """
    Recovers the structural shocks from smoothed state estimates.
    
    Parameters:
      A, B, C: Model matrices from the state equation.
      x_smooth: Smoothed state estimates (n_state x T).
      
    Returns:
      shocks: Recovered shocks (n_shocks x T).
    """
    T = x_smooth.shape[1]
    n_state = A.shape[0]
    nz = B.shape[1]  # number of shocks
    
    shocks = np.zeros((nz, T))
    # Compute pseudoinverse of B:
    B_pinv = np.linalg.pinv(B)
    
    # For t = 1,...,T-1 (since t=0 has no t-1)
    for t in range(1, T):
        # Reshape to column vectors:
        x_t = x_smooth[:, t].reshape(-1, 1)
        x_tm1 = x_smooth[:, t - 1].reshape(-1, 1)
        # Recover shock:
        shock_t = B_pinv @ (x_t - A @ x_tm1 - C)
        shocks[:, t] = shock_t.flatten()
    
    return shocks


# -------------------------------
# Define the historical_decomposition function
# -------------------------------
def historical_decomposition(A, C, B, y0, Z_hist):
    """
    Compute historical decomposition for a linear system:
        y(t) = A @ y(t-1) + C + B @ z(t),
    given historical shocks Z_hist of shape (nz, T).

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Transition matrix mapping y(t-1) to y(t).
    C  : np.ndarray, shape (n,)
        Vector of constants.
    B : np.ndarray, shape (n, nz)
        Matrix mapping shocks z(t) to y(t).
    y0     : np.ndarray, shape (n,)
        Initial condition y(0).
    Z_hist : np.ndarray, shape (nz, T)
        Historical shocks for T periods (t=1..T).

    Returns
    -------
    HD : np.ndarray, shape (nz, n, T)
        HD[s, i, t] gives the contribution of shock s
        to variable i at time t, for t=1..T.
    deterministic_part : np.ndarray, shape (n, T)
        Contribution from initial conditions and the constant term,
        for each variable and time.
    """

    n = A.shape[0]
    nz, T = Z_hist.shape

    # Ensure C is flat
    C = C.flatten()

    # Array for partial contributions of each shock
    HD = np.zeros((nz, n, T))

    # Also store the deterministic contribution (no shocks)
    deterministic_part = np.zeros((n, T))

    # (1) Compute the pure deterministic path (zero shocks)
    y_prev = y0.copy()
    for t in range(T):
        y_this = A @ y_prev + C  # no shock term here
        deterministic_part[:, t] = y_this
        y_prev = y_this

    # (2) For each shock, simulate the path when only that shock is active.
    for s in range(nz):
        # Set up partial shocks: only shock s is active, the rest are zero.
        partial_shocks = np.zeros((nz, T))
        partial_shocks[s, :] = Z_hist[s, :].copy()

        y_prev = y0.copy()
        for t in range(T):
            y_this = A @ y_prev + C + B @ partial_shocks[:, t]
            # The contribution of shock s at time t is the difference from the deterministic path:
            partial_contribution = y_this - deterministic_part[:, t]
            HD[s, :, t] = partial_contribution
            y_prev = y_this

    return HD, deterministic_part



def compute_unconditional_moments(G1_sol, C_sol, impact, Sigma_z):
    """
    Computes the unconditional mean and covariance for the linear system:
    
        y(t) = G1_sol * y(t-1) + C_sol + impact * z(t),
    
    where z(t) ~ N(0, Sigma_z).

    Parameters
    ----------
    G1_sol : np.ndarray, shape (n, n)
        Transition matrix for y(t-1) => y(t).
    C_sol  : np.ndarray, shape (n,) or (n,1)
        Constant vector for the system (flattened if needed).
    impact : np.ndarray, shape (n, nz)
        Mapping from shocks z(t) to y(t).
    Sigma_z : np.ndarray, shape (nz, nz)
        Covariance matrix of the shock vector z(t).

    Returns
    -------
    mu_y : np.ndarray, shape (n,)
        Unconditional mean of y(t).
    Sigma_y : np.ndarray, shape (n, n)
        Unconditional covariance of y(t).
    """

    # Ensure C_sol is a 1D array
    C_sol = C_sol.flatten()

    # 1) Compute the unconditional mean: mu_y = (I - G1_sol)^{-1} * C_sol
    n = G1_sol.shape[0]
    I_n = np.eye(n)
    # We'll check if (I - G1_sol) is invertible
    M = I_n - G1_sol
    if np.linalg.matrix_rank(M) < n:
        raise ValueError("Matrix (I - G1_sol) is singular => can't compute unconditional mean.")
    
    mu_y = np.linalg.inv(M) @ C_sol

    # 2) Compute the unconditional covariance by solving the discrete Lyapunov equation:
    #    Sigma_y = G1_sol * Sigma_y * G1_sol' + impact * Sigma_z * impact'
    Q = impact @ Sigma_z @ impact.T
    # Solve for Sigma_y
    Sigma_y = solve_discrete_lyapunov(G1_sol, Q)

    return mu_y, Sigma_y



def print_mean_cov(mu_y, Sigma_y, varnames=None):
    """
    Print the unconditional mean (mu_y) and covariance (Sigma_y) in a formatted table
    using 'tabulate'.
    
    Parameters
    ----------
    mu_y : np.ndarray, shape (n,)
        Unconditional mean of each variable.
    Sigma_y : np.ndarray, shape (n, n)
        Unconditional covariance matrix of the variables.
    varnames : list of str, optional
        List of variable names of length n. If not provided, uses Var0..Var(n-1).
    """
    n = len(mu_y)
    if varnames is None:
        varnames = [f"Var{i}" for i in range(n)]
    
    # 1) Print a one-column table for the unconditional mean
    #    e.g.  Var   Mean
    table_mean = [["Variable", "Mean"]]
    for i in range(n):
        table_mean.append([varnames[i], f"{mu_y[i]:.4f}"])
    
    print("=== Unconditional Mean ===")
    print(tabulate(table_mean, headers="firstrow", tablefmt="pretty"))
    print()
    
    # 2) Print a 2D table for the covariance matrix
    #    The top row is ["Var\\Var", var1, var2, ...]
    #    Each subsequent row is [var_i, Sigma_y[i,0], Sigma_y[i,1], ...]
    header = ["Var \\ Var"] + varnames
    table_cov = [header]
    for i in range(n):
        row = [varnames[i]]
        for j in range(n):
            row.append(f"{Sigma_y[i,j]:.4f}")
        table_cov.append(row)
    
    print("=== Unconditional Covariance Matrix ===")
    print(tabulate(table_cov, headers="firstrow", tablefmt="pretty"))