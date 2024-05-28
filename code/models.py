"""
Models are built up from individual reactions. Reactions should fix their parameters and array indices they act on at compile time,
on the basis of given parameters and of a list of substrates.
"""

import itertools

import numpy as onp
import jax.numpy as jnp
from jax import jit, vmap
import diffrax
from numpy.polynomial import Polynomial
#from orthax import Polynomial

import pandas as pd


def find_index(all_substrates, substrate):
    s = onp.unique([c for c in all_substrates])
    i = onp.searchsorted(s, substrate)
    assert(s[i] == substrate)
    return i


def N_substrates(all_substrates):
    return len(onp.unique([c for c in all_substrates]))


def hill_catalysis_reaction(all_substrates="", params=dict(k=1, K=1, n=2,r="B", p="b", c="g")):
    if params["r"]:
        i = find_index(all_substrates, params["r"])
    if params["p"]:
        j = find_index(all_substrates, params["p"])
    k = find_index(all_substrates, params["c"])
    N = N_substrates(all_substrates)
    def reaction(t, x, _):
        C = x[k]
        R = params['k']/(1+ (params['K']/C)**params['n'])
        f = jnp.zeros(N)
        if params["r"]:
            R = R * x[i]
            f = f.at[i].set(-R)
        if params["p"]:
            f = f.at[j].set(R)
        return f

    return reaction

def linear_catalysis_reaction(all_substrates="", params=dict(k=1,r="B", p="b", c="g")):
    if params["r"]:
        i = find_index(all_substrates, params["r"])
    if params["p"]:
        j = find_index(all_substrates, params["p"])
    if params["c"]:
        k = find_index(all_substrates, params["c"])
    N = N_substrates(all_substrates)
    def reaction(t, x, _):
        R = params['k']
        f = jnp.zeros(N)
        if params['c']:
            R = R *x[k]
        if params["r"]:
            R = R * x[i]
            f = f.at[i].set(-R)
        if params["p"]:
            f = f.at[j].set(R)
        return f

    return reaction



def add_reactions(reactions):
    @jit
    def f(t, x, _):
        return jnp.sum(jnp.array([reaction(t,x,_) for reaction in reactions]), axis=0)
    
    return f

def insulin_secretion_reaction(all_substrates="", params=dict(k=1, K=1, n=2)):
    '''
    Here we hard-code a lot for now.
    '''
    i = find_index(all_substrates, "G")
    j = find_index(all_substrates, "I")
    k = find_index(all_substrates, "B")
    l = find_index(all_substrates, "M")

    N = N_substrates(all_substrates)
    def reaction(t, x, _):
        C = x[i]
        R = params['k']*(C**params['n'])/(C**params['n'] + params['K']**params['n'])
        R = R*x[k]*x[l]
        f = jnp.zeros(N)
        f = f.at[j].set(R)
        return f

    return reaction

def flux(all_substrates="", params=dict(t0=0, t1=10000, f=10000, s="")):
    i = find_index(all_substrates, params['s'])
    N = N_substrates(all_substrates)

    def reaction(t, x, _):
        R = jnp.where(jnp.logical_and( t >= params['t0'], t <=params['t1']), params['f'], 0.0)

        f = jnp.zeros(N)
        f = f.at[i].set(R)
        return f
    
    return reaction

def integrator(f, tol=1e-3, t1=50, dt=1, max_steps=10000):
    term = diffrax.ODETerm(f)
    solver = diffrax.Kvaerno5()
    sc = diffrax.PIDController(rtol=tol, atol=tol)
 
    def g(x0):
        t = jnp.arange(0, t1+dt/2, dt)
        dt0 = dt
        saveat = diffrax.SaveAt(ts=t)
        return diffrax.diffeqsolve(term, solver, t[0], t[-1], dt0, x0, saveat=saveat, stepsize_controller=sc, max_steps=max_steps).ys

    return jit(g)


def model(reactions, max_steps=10000,t1=50,tol=1e-3, dt=1):
    f = add_reactions(reactions)
    return integrator(f, max_steps=max_steps,t1=t1,tol=tol, dt=dt)

def map_substrates(substrates):
    all_substrates = onp.unique([key for key in substrates])
    x = jnp.zeros(N_substrates(all_substrates))

    for i,s in enumerate(all_substrates):
        x = x.at[i].set(substrates[s])

    return x

def find_index(substrates, substrate):
    all_substrates = onp.unique([key for key in substrates])
    i = onp.where(all_substrates == substrate)[0]
    return i


def short_timescale_model(params, flux=0):
    reactions = []


def hill(G_0, n,k):
    return [Polynomial((0,)*n + (k,)), Polynomial((G_0**n,) + (0,)*(n-1) + (1,))]

def hill_plus_one(G_0, n,k):
    return [Polynomial((0,)*n + (k,)) + Polynomial((G_0**n,) + (0,)*(n-1) + (1,)),  Polynomial((G_0**n,) + (0,)*(n-1) + (1,))]

def evaluate(f, X):
    return f[0](X) / f[1](X)

def get_f(params):
    return hill(params["f_K"], params["f_n"],params["β"])

def get_r(params):
    return hill_plus_one(params["g_K"], params["g_n"],params["g_k"])


def final_poly(params):
    #f and r should both be [p_numerator, p_denominator]
    #1/r(G) = fraction of active beta cells. f(G) = secretion function x number of beta cells
    m_0 = params["m_0"]
    M = params["M"]
    S_E = params["S_E"]
    S_I = params["S_I"]
    I_0 = params["I_0"]
    F_I = params["F_I"] if "F_I" in params else 0.0
    G = Polynomial([0,1])
    f = get_f(params)
    r = get_r(params)
    
    return (m_0*f[1]*r[0]**2 + M*(f[0]*r[1] + I_0*f[1]*r[0]+F_I*r[0]*f[1])*r[0])*f[1] - \
        G*(S_I*(f[0]*r[1]+F_I*r[0]*f[1]) + S_E*f[1]*r[0])*((I_0+F_I)*r[0]*f[1] + r[1]*f[0])

def positive_roots(p):
    r = p.roots()
    arg = onp.where(onp.logical_and(onp.abs(r.imag) < 1e-14, r.real > 0))
    return r.real[arg]


def adaptive_beta(params):
    m_0 = params["m_0"]
    S_E = params["S_E"]
    S_I = params["S_I"]
    I_0 = params["I_0"]
    G_0 = params["G_0"]
    
    f = get_f(dict(params,β=1))
    r = get_r(params)
    
    f0 = evaluate(f, G_0) / evaluate(r, G_0)
    
    
    a = G_0*S_I*f0**2 
    b = (G_0*S_E + G_0*S_I*I_0)*f0
    c = (G_0*S_E*I_0 - m_0)
    
    return (-b + onp.sqrt(b**2 - 4*a*c))/(2*a)

def adaptive_m0(params):
    I = params["I_targ"]
    S_E = params["S_E"]
    S_I = params["S_I"]
    I_0 = params["I_0"]
    G_0 = params["G_0"]
    
    return G_0*(S_E+S_I*I)*(I_0+I)

def low_eq(params):
    G = positive_roots(final_poly(params))[0]
    r = get_r(params)

    return G, 1.0/evaluate(r, G)

def low_eq2(params):
    G = positive_roots(final_poly(params))[0]
    r = get_r(params)
    f = get_f(params)

    F = params['F_I'] if "F_I" in params else 0.0

    return onp.array([G, F + evaluate(f,G)/evaluate(r, G)])

def fast_model(params, meal_insulin_schedule=None, _b=None):
    t0 = 0

    if _b == 0.0: #special case: if there are no beta cells then error control algorithm is annoyed by I=0 always
        s = "G"
    else:
        s = "GI"

        #G_0 = params["G_0"]


    g=params['γ'] # we will come to this later
    if not _b == None:
        b = _b
        G_0, _ = low_eq(dict(params, β=b/g, g_k=0.0)) #manually set beta cell fraction instead of equilibrating reversible deactivation
        I_0 = evaluate(get_f(dict(params, β=b/g)), G_0)
    else:
        r = get_r(params)
        f = get_f(params)
        G_0, _ = low_eq(params)
        b = g*params['β'] / evaluate(r, G_0)
        I_0 = evaluate(f,G_0)/(evaluate(r, G_0))
    x = jnp.array((G_0, I_0) ).reshape((1,-1))

    #everything but meals
    HGP = hill_catalysis_reaction(all_substrates=s, params=dict(k=params["m_0"]/params["I_0"], p="G", n=-1, K=params["I_0"], c="I", r=None))
    if not b == 0.0:
        insulin_glucose_disposal = linear_catalysis_reaction(all_substrates=s, params=dict(k=params["S_I"], r="G", n=-1, c="I", p=None))
    glucose_effectiveness = linear_catalysis_reaction(all_substrates=s, params=dict(k=params["S_E"], r="G", n=-1, c=None, p=None))
    if not b == 0.0:
        insulin_secretion = hill_catalysis_reaction(all_substrates=s, params=dict(k=b, n=params['f_n'], K=params['f_K'],c="G",p="I",r=None))
        insulin_disposal = linear_catalysis_reaction(all_substrates=s, params=dict(k=g, r="I", p=None, c=None))
    if b == 0.0:
        base_reactions = [HGP, glucose_effectiveness]
    else:
        base_reactions = [HGP, insulin_glucose_disposal, glucose_effectiveness, insulin_secretion, insulin_disposal]


    t0 = 0
    for (t,  M, I) in meal_insulin_schedule:
        if b == 0.0:
            assert(I == 0.0)
        reactions = base_reactions + [ flux(all_substrates=s, params=dict(s="G", t0=0, t1=t, f=M)), flux(all_substrates=s, params=dict(s="I", t0=0, t1=t, f=I))]
        m = model(reactions, t1 = t-t0, dt=1/(60*24),tol=1e-2,max_steps=100000000)
        x = jnp.concatenate((x[:-1],m(x[-1])))
        t0 = t 

    all_t = jnp.arange(0, t0+1/(2*60*24), 1/(60*24))

    return all_t, x


def full_model(params, meal_schedule=None, days=None):
    t0 = 0

    s = "GI"

    G_0 = params["G_0"]
    r = get_r(params)
    f = get_f(params)


    b = params['β'] / evaluate(r, G_0)
    g=params['γ'] # we will come to this later
    x = jnp.array((G_0, evaluate(f,G_0)/(evaluate(r, G_0)))).reshape((1,-1))

    #everything but meals
    HGP = hill_catalysis_reaction(all_substrates=s, params=dict(k=params["m_0"], p="G", n=-1, K=params["I_0"], c="I", r=None))
    insulin_glucose_disposal = linear_catalysis_reaction(all_substrates=s, params=dict(k=params["S_I"], r="G", n=-1, c="I", p=None))
    glucose_effectiveness = linear_catalysis_reaction(all_substrates=s, params=dict(k=params["S_E"], r="G", n=-1, c=None, p=None))
    insulin_secretion = hill_catalysis_reaction(all_substrates=s, params=dict(k=b*g, n=params['f_n'], K=params['f_K'],c="G",p="I",r=None))
    insulin_disposal = linear_catalysis_reaction(all_substrates=s, params=dict(k=g, r="I", p=None, c=None))


    base_reactions = [HGP, insulin_glucose_disposal, glucose_effectiveness, insulin_secretion, insulin_disposal]


    for d in range(days):
        t0 = 0
        for (t,  M) in meal_schedule:
            reactions = base_reactions + [ flux(all_substrates=s, params=dict(s="G", t0=0, t1=t, f=M))]
            m = model(reactions, t1 = t-t0, dt=0.5)
            x = jnp.concatenate((x[:-1],m(x[-1])))
            t0 = t 

    all_t = jnp.arange(0, 24*days+0.25, 0.5)

    return all_t, x

def intermediate_rates(params, meal_insulin_schedule=None, resolution=0.01):
    r = hill(params["g_K"], params["g_n"],params["g_k"]*params["k_r"])
    deact = []
    react = []

    for active_frac in onp.arange(resolution, 1.0+resolution/2, resolution):
        b = params["beta_tot"]*active_frac
        t, x = fast_model(params, meal_insulin_schedule=meal_insulin_schedule, _b=b)
        G = x[:,0]
        deact.append(active_frac*onp.mean(evaluate(r, onp.array(G)))) #this only works because time is measured in days
        react.append((1.0-active_frac)*params["k_r"]) #this only works because time is measured in days

    return onp.arange(resolution, 1.0+resolution/2, resolution), onp.array(deact), onp.array(react)


def r_integral(params,  meal_insulin_schedule=None, S_I=None, beta=None):
    r = hill(params["g_K"], params["g_n"],params["g_k"]*params["k_r"])
    t, x = fast_model(dict(params,S_I=S_I), meal_insulin_schedule=meal_insulin_schedule, _b=beta)
    G = x[:,0]

    return onp.trapz((evaluate(r, onp.array(G))))

def r_beta_scan(params, S_I, beta_range, beta_samples, meal_insulin_schedule):
    #r_array = onp.zeros(beta_samples)
    output = []
    for i, beta in enumerate(onp.linspace(beta_range[0],beta_range[1], beta_samples)):
        try:
          r = r_integral(params, meal_insulin_schedule=meal_insulin_schedule, S_I=S_I, beta=beta)
        except:
          r = onp.nan
        output.append(dict(r=r, S_I=S_I, beta=beta))
    return pd.DataFrame(output)



def fast_model_2(params):
    t0 = 0

    s = "GI"
    g=params['γ'] # we will come to this later
    #G_0, _ = fp.low_eq(dict(params, β=b/g, g_k=0.0)) #manually set beta cell fraction instead of equilibrating reversible deactivation
    #I_0 = fp.evaluate(fp.get_f(dict(params, β=b/g)), G_0)

    #G_0 = params["G_0"]


    #everything but meals
    HGP = hill_catalysis_reaction(all_substrates=s, params=dict(k=params["m_0"]/params["I_0"], p="G", n=-1, K=params["I_0"], c="I", r=None))
    glucose_effectiveness = linear_catalysis_reaction(all_substrates=s, params=dict(k=params["S_E"], r="G", n=-1, c=None, p=None))
    insulin_disposal = linear_catalysis_reaction(all_substrates=s, params=dict(k=g, r="I", p=None, c=None))


    @jit
    def fast(a, S_I, b, G_0, I_0):
        I = params['F_I']*params['γ']

        t0 = 0
        #G_0, _ = fp.low_eq(dict(params, β=b/g, g_k=0.0, S_I=S_I)) #manually set beta cell fraction instead of equilibrating reversible deactivation
        #G_0, I_0 = jax.pure_callback(fp.low_eq2, jnp.array([0.0,0.0]), dict(params, β=b/g, g_k=0.0, S_I=S_I))
        #I_0 = fp.evaluate(fp.get_f(dict(params, β=b/g, S_I=S_I)), G_0)
        #I_0 = jax.pure_callback( lambda x: fp.evaluate(fp.get_f(x), G_0), jnp.array([0.0]), dict(params, β=b/g, g_k=0.0, S_I=S_I))[0]
        #I_0 = (b/g)*((G_0)**params["f_n"]/ ((G_0)**params["f_n"] + (params["f_K"])**params["f_n"]))
        insulin_secretion = hill_catalysis_reaction(all_substrates=s, params=dict(k=b, n=params['f_n'], K=params['f_K'],c="G",p="I",r=None))
        
        insulin_glucose_disposal = linear_catalysis_reaction(all_substrates=s, params=dict(k=S_I, r="G", n=-1, c="I", p=None))
        base_reactions = [HGP, insulin_glucose_disposal, glucose_effectiveness, insulin_secretion, insulin_disposal]
        
        x = jnp.array((G_0, I_0) ).reshape((1,-1))
        meal_insulin_schedule = [(12/48,0.0, I),(13/48,a, I),(24/48,0.0, I),(25/48,a, I) ,(36/48,0.0, I), (37/48, a, I), (1.0,0, I)]

        for (t,  M, I) in meal_insulin_schedule:
            reactions = base_reactions + [ flux(all_substrates=s, params=dict(s="G", t0=0, t1=t, f=M)), flux(all_substrates=s, params=dict(s="I", t0=0, t1=t, f=I))]
            m = model(reactions, t1 = t-t0, dt=1/(60*24),tol=1e-8,max_steps=1000)
            X = m(x[-1])
            x = jnp.concatenate((x[:-1],X))
            t0 = t 

        all_t = jnp.arange(0, t0+1/(2*60*24), 1/(60*24))

        return all_t, x
    
    return fast

def r_table(params, a_list=None,F_I=0.0,beta_range=None, SI_range=None,SI_samples=None, beta_samples=1000):

    model = fast_model_2(dict(params, F_I=F_I/params['γ']))
    v_model = vmap(model, in_axes=(0,None,None,None, None))
    output = []
    r = get_r(params)
    for _beta in onp.linspace(beta_range[0],beta_range[1], beta_samples):
        for _S_I in onp.linspace(SI_range[0],SI_range[1], SI_samples):
            if _beta == 0.0 and F_I == 0.0: #trick to avoid issues with error control when I=0.0
                beta = 1.0 
                S_I = 0.0
            else:
                beta = _beta
                S_I = _S_I
            G_0, I_0 = low_eq2(dict(params, β=beta/params['γ'], S_I=S_I, g_k=0.0, F_I=F_I/params['γ']))
            for i, a in enumerate(a_list):
                t,x = model(a, S_I, beta, G_0, I_0)

                G = onp.array(x[:,0])
                out = onp.trapz(evaluate(r, G), x=t) -1
                output.append(pd.DataFrame(dict(a=a, S_I=_S_I, beta=_beta, r=out, G_0=G_0), index=[0]))


    return pd.concat(output, axis=0, ignore_index=True)



def resample(B, r, B0):
    arg = onp.where(B < B0)
    return B[arg]/B0, r[arg]


def numerical_roots(beta, r, B0, params, S_I):
    B, r = resample(beta, r, B0)

    BB = onp.linspace(B[0], B[-1], 10000)
    X = -BB*onp.interp(BB, B, r)  + (1-BB)
    b_roots = B0*BB[onp.where(X[:-1]*X[1:] < 0)]
    if len(b_roots) < 1:
        b_roots = [B0]
    return [low_eq2(dict(params, β=b/params['γ'], S_I=S_I, g_k=0.0))[0] for b in b_roots][::-1]

def SI_beta_scan(params, SI_range=None, beta_range=None, SI_samples=10,beta_samples=1000, G_0=80, a=None, table_beta_samples=3000):
    r = get_r(params)
    m_0 = params["m_0"]
    M = 0
    S_E = params["S_E"]
    I_0 = params["I_0"]


    G = onp.zeros((SI_samples, beta_samples))
    classification = onp.zeros((SI_samples, beta_samples), dtype=int) #0 = 1 low min, 1 = 2 min, 2 = 1 high min
    def c(g, roots):
        if g <= roots[0]:
            return 0
        elif g <= roots[1]:
            return 1
        else:
            assert(g > roots[-1])
            return 2
        
    SI = onp.linspace(SI_range[0], SI_range[1], SI_samples)
    if beta_range[0] > 0:
        betas = onp.linspace(beta_range[0], beta_range[1], beta_samples)
    else:
        betas = onp.linspace(beta_range[0], beta_range[1], beta_samples+1)[1:]
    stuff = r_table(params, a_list = [a], SI_range=SI_range, SI_samples=SI_samples, beta_range=[beta_range[0]/100,beta_range[1]], beta_samples=table_beta_samples)
    
    for i, S_I in enumerate(SI):
        s = stuff[stuff['S_I']==S_I]
        r = s['r'].values   
        beta = s['beta'].values
        roots = onp.zeros((len(betas),3))
        boundary_roots = onp.full(4,-onp.inf)
        boundary_roots[0] = onp.inf

        for j, B0 in enumerate(betas):
            R = numerical_roots(beta, r, B0, params, S_I)
            try:
                roots[j] = R
            except Exception as e:
                print(e)
                print(R)
                return dict(beta=beta, r=r, B0=B0, params=params, S_I=S_I)
            if len(R) == 3:
                boundary_roots[0] = onp.min((boundary_roots[0], roots[j][0]))
                boundary_roots[1] = onp.max((boundary_roots[1], roots[j][0]))
                boundary_roots[2] = onp.max((boundary_roots[2], roots[j][1]))
                boundary_roots[3] = onp.max((boundary_roots[3], roots[j][2]))
        #roots = num_roots_scan(dict(params, S_I=S_I)) #this characterizes boundaries of region of bistability. strictly, resolution not controlled
        #max_beta = adaptive_beta(dict(params,S_I=S_I, G_0=G_0))
        roots = onp.where(roots>G_0, roots, G_0)
        for j, beta in enumerate(betas):
            g = roots[j][0]
            G[i,j] = g 
            try:
                classification[i,j] = c(g, boundary_roots)
            except Exception as e:
                print(g, boundary_roots)
                print(e)
                return dict(g=g, betas=betas,roots=roots, boundary_roots=boundary_roots, S_I=S_I, B0=beta, r=r, all_beta=s['beta'].values)


    return SI, betas, G, classification

def full_eq(params):
    G, frac = low_eq(params)
    B = params['beta_tot']*frac/params['γ']
    b = params['beta_tot']/params['γ']-B
    f = get_f(params)
    
    I = evaluate(f, G)*frac
    
    return map_substrates({"B":B, "b":b, "G":G, "I":I, "g":G, "M":1})
    
    
    


def full_model_day(params, a, I):
    t0 = 0

    s = "BbGIgM"
    g=params['γ'] # we will come to this later
    #G_0, _ = fp.low_eq(dict(params, β=b/g, g_k=0.0)) #manually set beta cell fraction instead of equilibrating reversible deactivation
    #I_0 = fp.evaluate(fp.get_f(dict(params, β=b/g)), G_0)

    #G_0 = params["G_0"]


    #everything but meals
    HGP = hill_catalysis_reaction(all_substrates=s, params=dict(k=params["m_0"]/params["I_0"], p="G", n=-1, K=params["I_0"], c="I", r=None))
    glucose_effectiveness = linear_catalysis_reaction(all_substrates=s, params=dict(k=params["S_E"], r="G", n=-1, c=None, p=None))
    insulin_glucose_disposal = linear_catalysis_reaction(all_substrates=s, params=dict(k=params["S_I"], r="G", n=-1, c="I", p=None))
    insulin_secretion = insulin_secretion_reaction(all_substrates=s, params=dict(k=params['γ'], n=params['f_n'], K=params['f_K']))
    insulin_disposal = linear_catalysis_reaction(all_substrates=s, params=dict(k=params['γ'], r="I",p=None,c=None))

    dediff = hill_catalysis_reaction(all_substrates=s, params=dict(k=params['k_r']*params['g_k'], n=params['g_n'], K=params['g_K'], r="B", p="b", c="G"))
    rediff = linear_catalysis_reaction(all_substrates=s, params=dict(k=params['k_r'], r="b", p="B",c=None))
    death1 = hill_catalysis_reaction(all_substrates=s, params=dict(k=params['k_d'], n=2, K=4000, r="B", c="G",p=None))
    death2 = hill_catalysis_reaction(all_substrates=s, params=dict(k=params['k_d'], n=2, K=4000, r="b", c="G",p=None))

    base_reactions = [HGP, insulin_glucose_disposal, glucose_effectiveness, insulin_secretion, insulin_disposal, dediff, rediff, death1, death2]
    meal_insulin_schedule = [(12/48,0.0, I),(13/48,a, I),(24/48,0.0, I),(25/48,a, I) ,(36/48,0.0, I), (37/48, a, I), (1.0,0, I)]
    @jit
    def M(x):        
        t0 = 0.0
        for (t,  M, I) in meal_insulin_schedule:
            reactions = base_reactions + [ flux(all_substrates=s, params=dict(s="G", t0=0, t1=t, f=M)), flux(all_substrates=s, params=dict(s="I", t0=0, t1=t, f=I))]
            m = model(reactions, t1 = t-t0, dt=1/(60*24),tol=1e-6,max_steps=1000)
            X = m(x[-1])
            x = jnp.concatenate((x[:-1],X))
            t0 = t 

        all_t = jnp.arange(0, t0+1/(2*60*24), 1/(60*24))

        return all_t, x
    
    return M
                          
                          
def full_simulation(params, pre_sugar_days=0, sugar_days=0, post_sugar_days=0, insulin_days=0, post_insulin_days=0, low_sugar=0, high_sugar=0, insulin=0):
    normal_model = full_model_day(params, low_sugar, 0.0)
    sugar_model = full_model_day(params, high_sugar, 0.0)
    insulin_model = full_model_day(params, low_sugar, insulin)
    s = "BbGIgM"
    x = full_eq(params).reshape((1,-1))
    g = [x[-1,1], ]
    B = [x[-1, 0], ]
    b = [x[-1,-1], ]
    g = [x[-1,1], ]
    I = [x[-1,find_index(s, "I")], ]
    
    
            
    for d in range(pre_sugar_days):
        t, x = normal_model(x[-1,:].reshape((1,-1)))
        g.append(onp.mean(x[:,1]))
        B.append(onp.mean(x[:,0]))
        b.append(onp.mean(x[:,-1]))
        I.append(onp.mean(x[:,find_index(s, "I")]))
                          
    for d in range(sugar_days):
        t, x = sugar_model(x[-1,:].reshape((1,-1)))
        g.append(onp.mean(x[:,1]))
        B.append(onp.mean(x[:,0]))
        b.append(onp.mean(x[:,-1]))
        I.append(onp.mean(x[:,find_index(s, "I")]))
   
    for d in range(post_sugar_days):
        t, x = normal_model(x[-1,:].reshape((1,-1)))
        g.append(onp.mean(x[:,1]))
        B.append(onp.mean(x[:,0]))
        b.append(onp.mean(x[:,-1]))
        I.append(onp.mean(x[:,find_index(s, "I")]))
   
    for d in range(insulin_days):
        t, x = insulin_model(x[-1,:].reshape((1,-1)))
        g.append(onp.mean(x[:,1]))
        B.append(onp.mean(x[:,0]))
        b.append(onp.mean(x[:,-1]))
        I.append(onp.mean(x[:,find_index(s, "I")]))
      
    for d in range(post_insulin_days):
        t, x = normal_model(x[-1,:].reshape((1,-1)))
        g.append(onp.mean(x[:,1]))
        B.append(onp.mean(x[:,0]))
        b.append(onp.mean(x[:,-1]))
        I.append(onp.mean(x[:,find_index(s, "I")]))
       
    return g, I, B, b, onp.arange(len(g))
                          
                          

                          
                          
                          

                          