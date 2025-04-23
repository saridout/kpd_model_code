"""
Models are built up from individual reactions.
Reactions should fix their parameters and array indices they act on at compile time,
on the basis of given parameters and of a list of substrates.
"""

import numpy as onp
import jax.numpy as jnp
from jax import jit
import diffrax
from numpy.polynomial import Polynomial
#from orthax import Polynomial

import pandas as pd


def find_index(all_substrates, substrate):
    """
    Args:
        all_substrates (string): string with each character being a substrate (e.g., "GI")
        substrate (string): single character identifying a substrate (e.g, "G")
    Returns:
        int: index of substrate in sorted list of all_substrates
    """
    s = onp.unique(list(all_substrates))
    i = onp.searchsorted(s, substrate)
    assert s[i] == substrate
    return i


def N_substrates(all_substrates):
    """
    Args:
        all_substrates (string): string with each character being a substrate (e.g., "GI")
    Returns:
        int: Number of substrates in the model.
    """
    return len(onp.unique([list(all_substrates)]))


def hill_catalysis_reaction(all_substrates, params):
    """
    Args:
        all_substrates (string): string with each character being a substrate (e.g., "GI")
        params (dict): dictionary of hill function parameters and identification of reactants
    Returns:
        function: A function that returns contribution to d[all concentrations]/dt,
            from a reaction which converts of r to p with rate k r c^n / (K^n + c^n)
    """
    assert "k" in params
    assert "K" in params
    assert "n" in params
    assert "c" in params
    assert ("r" in params) or ("p" in params)

    if "r" in params and params["r"]:
        i = find_index(all_substrates, params["r"])
    if "p" in params and params["p"]:
        j = find_index(all_substrates, params["p"])
    k = find_index(all_substrates, params["c"])
    N = N_substrates(all_substrates)
    def reaction(_t, x, _):
        C = x[k]
        R = params['k']/(1+ (params['K']/C)**params['n'])
        f = jnp.zeros(N)
        if "r" in params and params["r"]:
            R = R * x[i]
            f = f.at[i].set(-R)
        if "p" in params and params["p"]:
            f = f.at[j].set(R)
        return f

    return reaction

def linear_catalysis_reaction(all_substrates, params):
    """
    Args:
        all_substrates (string): string with each character being a substrate (e.g., "GI")
        params (dict): dictionary of rate constant k and identification of reactants
    Returns:
        function: A function that returns contribution to d[all concentrations]/dt,
            from a reaction which converts of r to p with rate k c r
    """
    assert "k" in params
    assert ("r" in params) or ("c" in params)
    assert ("r" in params) or ("p" in params)

    if "r" in params and params["r"]:
        i = find_index(all_substrates, params["r"])
    if "p" in params and params["p"]:
        j = find_index(all_substrates, params["p"])
    if "c" in params and params["c"]:
        k = find_index(all_substrates, params["c"])
    N = N_substrates(all_substrates)
    def reaction(_t, x, _):
        R = params['k']
        f = jnp.zeros(N)
        if 'c' in params and params['c']:
            R = R *x[k]
        if 'r' in params and params["r"]:
            R = R * x[i]
            f = f.at[i].set(-R)
        if 'p' in params and params["p"]:
            f = f.at[j].set(R)
        return f

    return reaction



def add_reactions(reactions):
    """
    Args:
        reactions (list): list of reaction functions which compute terms in dX/dt
    Returns:
        function: A function that computes the sum dX/dt of these terms.
    """
    @jit
    def f(t, x, _):
        return jnp.sum(jnp.array([reaction(t,x,_) for reaction in reactions]), axis=0)

    return f

def insulin_secretion_reaction(all_substrates, params):
    """
    Args:
        all_substrates (string): string with each character being a substrate (e.g., "GIBb")
        params (dict): dictionary of constants k, n, K
    Returns:
        function: A function that returns contribution to d[all concentrations]/dt,
            from insulin secretion at rate k B G^n / (G^n + K^n)
    """
    i = find_index(all_substrates, "G")
    j = find_index(all_substrates, "I")
    k = find_index(all_substrates, "B")
    l = find_index(all_substrates, "M")

    N = N_substrates(all_substrates)
    def reaction(_t, x, _):
        C = x[i]
        R = params['k']*(C**params['n'])/(C**params['n'] + params['K']**params['n'])
        R = R*x[k]*x[l]
        f = jnp.zeros(N)
        f = f.at[j].set(R)
        return f

    return reaction

def flux(all_substrates, params):
    i = find_index(all_substrates, params['s'])
    N = N_substrates(all_substrates)

    def reaction(t, _x, _):
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
        return diffrax.diffeqsolve(term, solver, t[0], t[-1], dt0, x0, saveat=saveat,
                                   stepsize_controller=sc, max_steps=max_steps).ys

    h = jit(g)
    h: callable #jit confuses pylint, it seems
    return h


def model(reactions, max_steps=10000,t1=50,tol=1e-3, dt=1):
    f = add_reactions(reactions)
    return integrator(f, max_steps=max_steps,t1=t1,tol=tol, dt=dt)

def map_substrates(substrates):
    all_substrates = onp.unique(list(substrates))
    x = jnp.zeros(N_substrates(all_substrates))

    for i,s in enumerate(all_substrates):
        x = x.at[i].set(substrates[s])

    return x

def hill(G_0, n,k):
    return [Polynomial((0,)*n + (k,)), Polynomial((G_0**n,) + (0,)*(n-1) + (1,))]

def hill_plus_one(G_0, n,k):
    return [Polynomial((0,)*n + (k,)) + Polynomial((G_0**n,) + (0,)*(n-1) + (1,)),
            Polynomial((G_0**n,) + (0,)*(n-1) + (1,))]

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

def g_poly(params, frac):
    #fixed active beta cell fraction
    m_0 = params["m_0"]
    M = params["M"]
    S_E = params["S_E"]
    S_I = params["S_I"]
    I_0 = params["I_0"]
    F_I = params["F_I"] if "F_I" in params else 0.0
    G = Polynomial([0,1])
    f = get_f(params)
    r = [1, frac]

    return (m_0*f[1]*r[0]**2 + M*(f[0]*r[1] + I_0*f[1]*r[0]+F_I*r[0]*f[1])*r[0])*f[1] - \
        G*(S_I*(f[0]*r[1]+F_I*r[0]*f[1]) + S_E*f[1]*r[0])*((I_0+F_I)*r[0]*f[1] + r[1]*f[0])

def eq_beta_frac(params, G):
    r = get_r(params)
    return 1/evaluate(r, G)

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
    c = G_0*S_E*I_0 - m_0

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

def all_eq(params):
    G = positive_roots(final_poly(params))
    r = get_r(params)
    frac = [1.0/evaluate(r, g) for g in G]

    return G, frac



def fast_model(params):
    s = "GI"
    g=params['γ'] # we will come to this later


    #everything but meals
    HGP = hill_catalysis_reaction(s, {"k": params["m_0"] / params["I_0"],
                                      "p": "G", "n": -1, "K": params["I_0"], "c": "I"})

    glucose_effectiveness = linear_catalysis_reaction(s, {"k": params["S_E"], "r": "G"})

    insulin_disposal = linear_catalysis_reaction(s, {"k": g, "r": "I"})

    @jit
    def fast(a, S_I, b, G_0, I_0):
        I = params['F_I']*params['γ']

        t0 = 0

        I_secretion = hill_catalysis_reaction(s,{"k": b, "n": params['f_n'],
                                                 "K": params['f_K'], "c": "G", "p": "I"})
        insulin_glucose_disposal = linear_catalysis_reaction(s, {"k": S_I, "r": "G", "c": "I"})

        base_reactions = [HGP, insulin_glucose_disposal,
                          glucose_effectiveness, I_secretion, insulin_disposal]

        x = jnp.array((G_0, I_0) ).reshape((1,-1))
        meal_insulin_schedule = [(12/48,0.0, I),(13/48,a, I),
                                 (24/48,0.0, I),(25/48,a, I),
                                 (36/48,0.0, I), (37/48, a, I), (1.0,0, I)]

        for (t,  M, I) in meal_insulin_schedule:
            reactions = base_reactions + [flux(s, {"s": "G", "t0": 0, "t1": t, "f": M}),
                              flux(s, {"s": "I", "t0": 0, "t1": t, "f": I})]
            m = model(reactions, t1 = t-t0, dt=1/(60*24),tol=1e-8,max_steps=1000)
            X = m(x[-1])
            x = jnp.concatenate((x[:-1],X))
            t0 = t

        all_t = jnp.arange(0, t0+1/(2*60*24), 1/(60*24))

        return all_t, x

    return fast

def r_table(params, a_list=None,F_I=0.0,beta_range=None,
            SI_range=None,SI_samples=None, beta_samples=1000):

    simulate_fast = fast_model(dict(params, F_I=F_I/params['γ']))
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

            G_0, I_0 = low_eq2(params | {"β": beta / params['γ'],
                    "S_I": S_I,
                    "g_k": 0.0,
                    "F_I": F_I / params['γ']})

            for a in a_list:
                t,x = simulate_fast(a, S_I, beta, G_0, I_0)

                G = onp.array(x[:,0])
                out = onp.trapz(evaluate(r, G), x=t) -1
                output.append(
                    pd.DataFrame({"a": a, "S_I": _S_I,
                                  "beta": _beta, "r": out, "G_0": G_0}, index=[0])
                )


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

def SI_beta_scan(params, SI_range=None, beta_range=None,
                 SI_samples=10,beta_samples=1000, G_0=80, a=None, table_beta_samples=3000):
    r = get_r(params)
    G = onp.zeros((SI_samples, beta_samples))

    #0 = 1 low min, 1 = 2 min, 2 = 1 high min
    classification = onp.zeros((SI_samples, beta_samples), dtype=int)
    def c(g, roots):
        if g <= roots[0]:
            return 0
        if g <= roots[1]:
            return 1
        assert g > roots[-1]
        return 2

    SI = onp.linspace(SI_range[0], SI_range[1], SI_samples)
    if beta_range[0] > 0:
        betas = onp.linspace(beta_range[0], beta_range[1], beta_samples)
    else:
        betas = onp.linspace(beta_range[0], beta_range[1], beta_samples+1)[1:]
    stuff = r_table(params, a_list = [a],
                    SI_range=SI_range, SI_samples=SI_samples,
                    beta_range=[beta_range[0]/100,beta_range[1]], beta_samples=table_beta_samples)

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
                return {"beta": beta, "r": r, "B0": B0, "params": params, "S_I": S_I}
            if len(R) == 3:
                boundary_roots[0] = onp.min((boundary_roots[0], roots[j][0]))
                boundary_roots[1] = onp.max((boundary_roots[1], roots[j][0]))
                boundary_roots[2] = onp.max((boundary_roots[2], roots[j][1]))
                boundary_roots[3] = onp.max((boundary_roots[3], roots[j][2]))

        roots = onp.where(roots>G_0, roots, G_0)
        for j, beta in enumerate(betas):
            g = roots[j][0]
            G[i,j] = g
            classification[i,j] = c(g, boundary_roots)


    return SI, betas, G, classification

def full_eq(params):
    G, frac = low_eq(params)
    B = params['beta_tot']*frac/params['γ']
    b = params['beta_tot']/params['γ']-B
    f = get_f(params)

    I = evaluate(f, G)*frac

    return map_substrates({"B":B, "b":b, "G":G, "I":I, "g":G, "M":1})

def full_model_day(params, a, _I):

    
    s = "BbGIgM"
    #everything but meals
    HGP = hill_catalysis_reaction(s, {"k":params["m_0"]/params["I_0"],
                                      "p":"G", "n":-1, "K":params["I_0"], "c":"I"})
    glucose_effectiveness = linear_catalysis_reaction(s, {"k":params["S_E"], "r":"G"})
    insulin_glucose_disposal = linear_catalysis_reaction(s, {"k":params["S_I"],
                                                             "r":"G", "n":-1, "c":"I"})
    insulin_secretion = insulin_secretion_reaction(s, {"k":params['γ'],
                                                       "n":params['f_n'],"K":params['f_K']})
    insulin_disposal = linear_catalysis_reaction(s, {"k":params['γ'], "r":"I"})

    dediff = hill_catalysis_reaction(s,
                                     {"k":params['k_r']*params['g_k'], "n":params['g_n'],
                                      "K":params['g_K'], "r":"B", "p":"b", "c":"G"})
    rediff = linear_catalysis_reaction(s, {"k": params['k_r'], "r": "b", "p": "B"})
    death1 = hill_catalysis_reaction(s, {"k": params['k_d'], "n": 2, "K": 4000, "r": "B", "c": "G"})
    death2 = hill_catalysis_reaction(s, {"k": params['k_d'], "n": 2, "K": 4000, "r": "b", "c": "G"})

    base_reactions = [HGP, insulin_glucose_disposal, glucose_effectiveness,
                      insulin_secretion, insulin_disposal, dediff, rediff, death1, death2]
    @jit
    def M(x):
        t0 = 0.0
        if callable(_I): #function for deciding insulin dose
            I = _I(params, s, x[-1])
        else: #hard-coded insulin dose
            I = _I
        meal_insulin_schedule = [(12/48,0.0, I),(13/48,a, I),
                         (24/48,0.0, I),(25/48,a, I),
                         (36/48,0.0, I), (37/48, a, I), (1.0,0, I)]

        for (t,  M, I) in meal_insulin_schedule:
            reactions = base_reactions + [ flux(s, {"s":"G", "t0":0, "t1":t, "f":M}),
                                          flux(s, {"s":"I", "t0":0, "t1":t, "f":I})]
            m = model(reactions, t1 = t-t0, dt=1/(60*24),tol=1e-6,max_steps=1000)
            X = m(x[-1])
            x = jnp.concatenate((x[:-1],X))
            t0 = t

        all_t = jnp.arange(0, t0+1/(2*60*24), 1/(60*24))

        return all_t, x

    return M


def full_simulation(params, pre_sugar_days=0, sugar_days=0, post_sugar_days=0,
                    insulin_days=0, post_insulin_days=0, low_sugar=0, high_sugar=0, insulin=0):
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

    def F_I(x):
        if callable(insulin):
            return insulin(params, s, x)
        else:
            return insulin

    F = [0.0, ]

    for _ in range(pre_sugar_days):
        _, x = normal_model(x[-1,:].reshape((1,-1)))
        g.append(onp.mean(x[:,1]))
        B.append(onp.mean(x[:,0]))
        b.append(onp.mean(x[:,-1]))
        I.append(onp.mean(x[:,find_index(s, "I")]))
        F.append(0.0)

    for _ in range(sugar_days):
        _, x = sugar_model(x[-1,:].reshape((1,-1)))
        g.append(onp.mean(x[:,1]))
        B.append(onp.mean(x[:,0]))
        b.append(onp.mean(x[:,-1]))
        I.append(onp.mean(x[:,find_index(s, "I")]))
        F.append(0.0)

    for _ in range(post_sugar_days):
        _, x = normal_model(x[-1,:].reshape((1,-1)))
        g.append(onp.mean(x[:,1]))
        B.append(onp.mean(x[:,0]))
        b.append(onp.mean(x[:,-1]))
        I.append(onp.mean(x[:,find_index(s, "I")]))
        F.append(0.0)

    for _ in range(insulin_days):
        _, x = insulin_model(x[-1,:].reshape((1,-1)))
        g.append(onp.mean(x[:,1]))
        B.append(onp.mean(x[:,0]))
        b.append(onp.mean(x[:,-1]))
        I.append(onp.mean(x[:,find_index(s, "I")]))
        F.append(F_I(x[0,:]))

    for _ in range(post_insulin_days):
        _, x = normal_model(x[-1,:].reshape((1,-1)))
        g.append(onp.mean(x[:,1]))
        B.append(onp.mean(x[:,0]))
        b.append(onp.mean(x[:,-1]))
        I.append(onp.mean(x[:,find_index(s, "I")]))
        F.append(0.0)


    return onp.array(g), onp.array(I), onp.array(B), onp.array(b), onp.array(F), onp.arange(len(g))

def fasting_maximum_insulin(G_min):
    """
    Returns a function that decides the maximum insulin flux which will produce fasting glucose at least G_min.
    """

    def FI_func(params, s, x):
        β = x[find_index(s, "B")]

        I_0 = params["I_0"]
        m_0 = params["m_0"]
        S_I = params["S_I"]
        S_E = params["S_E"]

        fasting_I = (-I_0 - S_E/S_I + onp.sqrt((I_0 + S_E/S_I)**2 + 4*(m_0 / (G_min*S_I)- S_E*I_0/S_I)))/2.0
        γ = params['γ']

        f = get_f(params | {"β": 1})
        f_min = evaluate(f, G_min)

        return jnp.where(γ*fasting_I - γ*β*f_min >0, γ*fasting_I - γ*β*f_min, 0.0)

    return FI_func


def fasting_max_insulin_timescale(params):
    """
    Recovery timescale under insulin treatment which fixes G at G_min.
    """
    G_roots = positive_roots(final_poly(params))
    assert len(G_roots) == 3 #if no bistablity, what are you doing?
    #check the order is right while we're at it
    assert G_roots[2] >= G_roots[1]
    assert G_roots[1] >= G_roots[0]

    frac_f = eq_beta_frac(params, G_roots[0])
    frac_0 = eq_beta_frac(params, G_roots[2])

    G_min = onp.arange(85, G_roots[0],0.5)

    
    h = params["k_r"]
    r = params["k_r"]*evaluate(get_r(params), G_min)

    τ = (1/r)*onp.log((h-r*frac_0)/(h-r*frac_f))

    return G_min, τ 

def fasting_approx_frac_schedule(params, G_min):
    G_roots = positive_roots(final_poly(params))
    frac_f = eq_beta_frac(params, G_roots[0])
    frac_0 = eq_beta_frac(params, G_roots[2])
    
    h = params["k_r"] #in the paper's notation, this is K_RE
    r = params["k_r"]*evaluate(get_r(params), G_min) #in the paper's notation, this is k_IN g(G) + k_RE

    τ = (1/r)*onp.log((h-r*frac_0)/(h-r*frac_f))

    t = onp.arange(0, τ+1)

    beta_frac = (1/r)*(onp.exp(-r*t)*(r*frac_0 - h) + h)

    return t, beta_frac

def fasting_approx_insulin_schedule(params, G_min):
    t, β = fasting_approx_frac_schedule(params, G_min)
    γ = params["γ"]

    f = evaluate(get_f(params), G_min)


    I_0 = params["I_0"]
    m_0 = params["m_0"]
    S_I = params["S_I"]
    S_E = params["S_E"]

    fasting_I = (-I_0 - S_E/S_I + onp.sqrt((I_0 + S_E/S_I)**2 + 4*(m_0 / (G_min*S_I)- S_E*I_0/S_I)))/2.0
    
    return t, γ*fasting_I - γ*β*f
    

    