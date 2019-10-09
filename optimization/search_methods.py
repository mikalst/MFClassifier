#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of:
    bactracking linesearch
    bisection linesearch
    zoom
    steepest descent
    bfgs
    fletcher-reeves
"""

import numpy as np


def backtracking_linesearch(f, g, x_k, p_k, g_k, verbosity=0):
    """Backtracking linesearch
    
    Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x_k : initial point.
    p_k : direction of search.
    g_k : gradient evaluated at inital point.
    
    Returns
    ----------
    x_k+1 : point satisfying sufficient decrease, or point at which the linesearch 
            was terminated
    success : Boolean value indicating the convergence of the linesearch."""
    
    f0 = f(x_k)
    alpha = 1
    c1 = 0.2

    sd = False
    while(not sd):
        alpha *= 0.5
        sd = f(x_k + alpha * p_k) <= f0 + c1 * alpha * g_k.T@p_k
    
    if (np.max(np.abs(alpha * p_k / x_k)) < 1.1E-8):
        if verbosity>=2:
            print("Backtracking did not converge")
        return x_k + alpha*p_k, False
    
    return x_k + alpha*p_k, True


def output_linesearch(alpha_j, f_j, g_j_norm):
    print("alpha = {:2f}".format(alpha_j),
          ", j = {:.3e}".format(f_j),
          ", dnorm = {:.3}".format(g_j_norm), sep="")


def zoom(f, g, x_k, p_k, alpha_lo, alpha_hi, c1, c2, verbosity=0):
    """Zoom
    
        Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x_k : initial point.
    p_k : direction of search.
    alpha_lo: lower value of alpha.
    alpha_hi: upper value of alpha.
    c1 : 1st parameter of the wolfe conditions.
    c2 : 2nd parameter of wolfe conditions. 
    
    Returns
    ----------
    x_k+1 : point satisfying strong wolfe, or point at which the linesearch 
            was terminated
    success : Boolean value indicating the convergence of the linesearch."""
    
    f0 = f(x_k)
    g0 = g(x_k)
        
    while True:
        
        alpha_i = (alpha_lo + alpha_hi)/2

        if (np.abs(alpha_lo - alpha_hi) < 1e-5):
            if verbosity>=2:
                print("Zoom did not converge")
            return x_k, False
        
        f_i = f(x_k + alpha_i*p_k)
        
        # Check if sufficient decrease condition satisfied
        if (f_i > f0 + c1*alpha_i*g0.dot(p_k)) or f_i >= f(x_k + alpha_lo * p_k):
            alpha_hi = alpha_i

        else:
            g_i = g(x_k + alpha_i * p_k)
            
            # Check if curvature condition is satisfied
            if np.abs(g_i.dot(p_k)) <= -c2*g0.dot(p_k):
                if verbosity>=2:
                    output_linesearch(alpha_i, f_i, np.linalg.norm(g_i))
                return x_k + alpha_i * p_k, True
            
            if g_i.dot(p_k)*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
                
            alpha_lo = alpha_i


def linesearch(f, g, x_k, p_k, c1, c2, wolfe='s', verbosity=0):
    """Linesearch
    
        Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x_k : initial point.
    p_k : direction of search.
    c1 : 1st parameter of the wolfe conditions.
    c2 : 2nd parameter of wolfe conditions. 
    wolfe : specification of wolfe conditions, {s, w}
    verbosity : level of logging
    
    Returns
    ----------
    x_k+1 : point satisfying strong wolfe, or point at which the linesearch 
            was terminated
    success : Boolean value indicating the convergence of the linesearch."""

    alpha_0 = 0
    alpha_max = np.inf
    alpha_i = 1
    
    f0 = f(x_k)
    g0 = g(x_k)
    
    f_last = f0
    alpha_last = alpha_0
    i = 1
    
    #Check if p_k is a valid direction. 
    if not(g(x_k).dot(p_k) < 0):
        if verbosity >= 2:
            print("Not a descent direction")
        return x_k, False
    
    while True:
        
        if (alpha_i < 1E-3):
            print("Very small step, consider scaling gradient")
        
        f_i = f(x_k + alpha_i * p_k)
        
        if f_i > f0 + c1*alpha_i*g0.dot(p_k) or (f_i >= f_last and i > 1):
            return zoom(f, g, x_k, p_k, alpha_last, alpha_i, c1, c2, verbosity)
        
        g_i = g(x_k + alpha_i * p_k)
        
        g_i_dot_p_k = g_i.dot(p_k)
        
        if np.abs(g_i_dot_p_k) <= -c2*g0.dot(p_k):
            if verbosity>=2:
                output_linesearch(alpha_i, f_i, np.linalg.norm(g_i))
            return x_k + alpha_i * p_k, True
        
        if g_i_dot_p_k >= 0:
            if wolfe=='s':
                return zoom(f, g, x_k, p_k, alpha_i, alpha_last, c1, c2, verbosity)
            
            elif wolfe=='w':
                if verbosity>=2:
                    output_linesearch(alpha_i, f_i, np.linalg.norm(g_i))
                return x_k + alpha_i * p_k, True
            
            else:
                print('Wolfe condition should be {\'s\', \'w\'}, \ndefaulting to strong ...')
                return zoom(f, g, x_k, p_k, alpha_i, alpha_last, c1, c2, verbosity)
        
        alpha_last = alpha_i
        if alpha_max == np.inf:
            alpha_i *= 2
        else:
            alpha_i = (alpha_i + alpha_max)/2
        i += 1


def steepest_descent(f, g, x0, TOL = 1e-3, max_iter = 99, verbosity=False):
    """Steepest descent algorithm for unconstrained minimization.
    
    Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x_k : initial point.
    TOL : maximum Euclidian-norm of gradient at local minimum.
    max_iter : maximum number of iterations.
    verbosity : flag for output.
    
    Returns
    ----------
    x_k : obtained minimum.
    iterations : iterations used.
    f_final : value of objective at local minimum.
    x_k_list : all iterates of x_k."""
    
    x_k = x0
    x_k_list = []
    g_k = g(x_k)
    p_k = -g_k
    
    if np.linalg.norm(g_k) < TOL:
        return x_k

    iterations = 1

    while True:
        x_k, success = backtracking_linesearch(f, g, x_k, p_k, g_k, verbosity)
        g_k = g(x_k)
        
        x_k_list.append(x_k)
        
        iterations += 1
        if np.linalg.norm(g_k) < TOL or iterations >= max_iter:
            break
        
        p_k = -g_k
    
    f_final = f(x_k)
        
    if verbosity>=1:
        print('SD j:{:.3e} -> {:.3e} in {} iterations.'.format(f(x0), f_final, iterations))
    
    return x_k, iterations, f_final, x_k_list


def bfgs(f, g, x0, TOL = 1e-3, max_iter = 99, linesearch_method = "ww", verbosity=0):
    """BFGS algorithm for unconstrained minimization.
    
    Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x : initial point.
    TOL : maximum Euclidian-norm of gradient at local minimum.
    max_iter : maximum number of iterations.
    linesearch_method : method to be used in linesearch. {bt, ww}
    verbosity : flag for output.
    
    Returns
    ----------
    x_k : obtained minimum.
    iterations : iterations used.
    f_final : value of objective at local minimum.
    x_k_list : all iterates of x_k."""
    
    I = np.identity(len(x0))
    iterations = 1
    reset_to_SD_counter = 0
    x_k = x0
    x_k_list = []
    g_k = g(x_k)
    H_k =  I

    while True:
        if verbosity>=2:
            print("iter = {}".format(iterations), end=", ")

        x_last = x_k
        g_last = g_k 
        
        p_k = - H_k@g_k
        
        if linesearch_method == "bt":
            x_k, ls_success = backtracking_linesearch(f, g, x_k, p_k, g_k, verbosity=verbosity)
            
        elif linesearch_method == "ww":
            x_k, ls_success = linesearch(f, g, x_k, p_k, 1e-4, 0.9, wolfe='w', verbosity=verbosity)
            
        x_k_list.append(x_k)
        
        g_k = g(x_k)
        
        iterations += 1
        if np.linalg.norm(g_k) < TOL or iterations >= max_iter:
            break
        
        #Reset to steepest descent either if our chosen direction is not a descent direction
        if not(ls_success): #(g_k.dot(p_k) / (np.linalg.norm(g_k) * np.linalg.norm(p_k))) < 1e-10
            reset_to_SD_counter += 1
            H_k = I
            continue
 
        y = g_k - g_last; 
        s = x_k - x_last;
        
        if (y.dot(s) <= 0):
            H_k = H_k
        else:
            rho = 1/y.dot(s)
            H_k = (I - rho*np.outer(s, y))@H_k@(I - rho*np.outer(y, s)) + rho*np.outer(s, s)
            
    f_final = f(x_k)
      
    if verbosity>=1:
        print('BFGS j:{:.3e} -> {:.3e} in {} iterations.'.format(f(x0), f_final, iterations))
    
    return x_k, iterations, f_final, x_k_list


def fletcher_reeves(f, g, x0, TOL = 1e-3, max_iter = 99, verbosity=0):
    """Fletcher-Reeves algorithm for unconstrained minimization.
    
    Parameters
    ----------
    f : objective, callable.
    g : gradient, callable.
    x : initial point.
    TOL : maximum Euclidian-norm of gradient at local minimum.
    max_iter : maximum number of iterations.
    verbosity : flag for output.
    
    Returns
    ----------
    x_k : obtained minimum.
    iterations : iterations used.
    f_final : value of objective at local minimum.
    x_k_list : all iterates of x_k."""
    
    x_k = x0
    
    x_k_list = []
    
    g_k = g(x_k)      
    p_k = -g_k
    
    iterations = 1
    reset_to_SD_counter = 0
    
    while True:
        if verbosity>=2:
            print("iter = {}".format(iterations), end=", ")
            
        x_last = x_k
        g_last = g_k
        p_last = p_k

        x_k, ls_success = linesearch(f, g, x_k, p_k, 1e-4, 0.90, wolfe='s', verbosity=verbosity)

        x_k_list.append(x_k)
        
        g_k = g(x_k)

        if np.linalg.norm(g_k) < TOL or iterations >= max_iter:
            break

        # Les i bok for hva en skal gjøre her, altså
        # Hva en skal gjøre hvis linjesøket ikke fører fram
        if not(ls_success):
            reset_to_SD_counter += 1
            iterations += 1
            p_k = -g_k
            continue

        beta_fr = g_k.dot(g_k)/(g_last.dot(g_last))
        beta_pr = g_k.dot(g_k - g_last) / (np.linalg.norm(g_last)**2)

        if beta_pr < -beta_fr:
            beta = -beta_fr
        elif np.abs(beta_pr) <= beta_fr:
            beta = beta_pr
        else:
            beta = beta_fr

        p_k = -g_k + beta*p_last

        iterations += 1
    
    f_final = f(x_k)
    
    if verbosity>=1:
        print('FR j:{:.3e} -> {:.3e} in {} iterations.'.format(f(x0), f_final, iterations))
    
    return x_k, iterations, f_final, x_k_list