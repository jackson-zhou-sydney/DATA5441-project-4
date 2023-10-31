
#### Numerical experiments for DATA5441 Project 4 (Jackson Zhou) ####

import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def erg_gen(n):
    return nx.erdos_renyi_graph(n, 1/n)

def gn_gen_1(n, alpha):
    return nx.gn_graph(n, kernel=lambda k: k ** alpha).to_undirected()

def gn_gen_2(n, C):
    return nx.gn_graph(n, kernel=lambda k: 1 if k == 1 else C * k).to_undirected()
  
#### Sub-linear regime ####

def f_erg(log_k, n):
    k = math.exp(log_k)
    return math.log(n) - 1 - math.lgamma(k - 1)

def f_0(log_k, n):
    k = math.exp(log_k)
    alpha = alpha_vals_sub[0]
    mu = mu_vals[0]
    return math.log(n) - alpha*math.log(k) - mu*((k**(1 - alpha))/(1 - alpha)) + ((mu**2)/2)*((k**(1 - 2*alpha))/(1 - 2*alpha))

def f_1(log_k, n):
    k = math.exp(log_k)
    alpha = alpha_vals_sub[1]
    mu = mu_vals[1]
    return math.log(n) + (mu**2 - 0.5)*math.log(k) - 2*mu*math.sqrt(k)

def f_2(log_k, n):
    k = math.exp(log_k)
    alpha = alpha_vals_sub[2]
    mu = mu_vals[2]
    return math.log(n) - alpha*math.log(k) - mu*((k**(1 - alpha) - 2**(1 - alpha))/(1 - alpha))

def f_3(log_k, n):
    k = math.exp(log_k)
    return math.log(4*n) - math.log(k*(k + 1)*(k + 2))

# Parameters
n = 10000
reps = 4
thresh_df = 0.5
thresh_d = 2
ms = 5
alpha_vals_sub = [0.35, 0.5, 0.65, 1]
mu_vals = [1.21, 1.32, 1.47]

# Experiments
gn_sub = [[gn_gen_1(n, alpha) for i in range(reps)] for alpha in alpha_vals_sub]
gn_sub_df = [[np.asarray(nx.degree_histogram(G)) for G in G_list] for G_list in gn_sub]
gn_sub_max_len = [max([len(df) for df in df_list]) for df_list in gn_sub_df]
gn_sub_df = [[np.pad(df, (0, gn_sub_max_len[i] - len(df)), mode="constant") for df in gn_sub_df[i]] for i in range(len(gn_sub))]
gn_sub_df = [np.mean(df_list, axis=0) for df_list in gn_sub_df]
gn_sub_d = [range(len(df)) for df in gn_sub_df]
gn_sub_inds = [[j for j in gn_sub_d[i] if gn_sub_df[i][j] >= thresh_df and gn_sub_d[i][j] >= thresh_d] for i in range(len(gn_sub))]
gn_sub_x = [[math.log(gn_sub_d[i][j]) for j in gn_sub_inds[i]] for i in range(len(gn_sub))]
gn_sub_y = [[math.log(gn_sub_df[i][j]) for j in gn_sub_inds[i]] for i in range(len(gn_sub))]

erg = [erg_gen(n) for i in range(reps)]
erg_df = [np.asarray(nx.degree_histogram(G)) for G in erg]
erg_max_len = max([len(df) for df in erg_df])
erg_df = [np.pad(df, (0, erg_max_len - len(df)), mode="constant") for df in erg_df]
erg_df = np.mean(erg_df, axis=0)
erg_d = range(len(erg_df))
erg_inds = [j for j in erg_d if erg_df[j] >= thresh_df and erg_d[j] >= thresh_d]
erg_x = [math.log(erg_d[j]) for j in erg_inds]
erg_y = [math.log(erg_df[j]) for j in erg_inds]

# Scatter plot
fig, ax = plt.subplots()
ax.set_ylim([-2, 8])

plt.plot(gn_sub_x[3], gn_sub_y[3], "or", ms=ms, label=f"α = {alpha_vals_sub[3]}")
plt.plot(gn_sub_x[2], gn_sub_y[2], "oy", ms=ms, label=f"α = {alpha_vals_sub[2]}")
plt.plot(gn_sub_x[1], gn_sub_y[1], "og", ms=ms, label=f"α = {alpha_vals_sub[1]}")
plt.plot(gn_sub_x[0], gn_sub_y[0], "ob", ms=ms, label=f"α = {alpha_vals_sub[0]}")
plt.plot(erg_x, erg_y, "ok", ms=ms, label="Poisson ERG")

plt.plot(gn_sub_x[3], [f_3(i, n) for i in gn_sub_x[3]], "--r", alpha=0.5)
plt.plot(gn_sub_x[3], [f_2(i, n) for i in gn_sub_x[3]], "--y", alpha=0.5)
plt.plot(gn_sub_x[3], [f_1(i, n) for i in gn_sub_x[3]], "--g", alpha=0.5)
plt.plot(gn_sub_x[3], [f_0(i, n) for i in gn_sub_x[3]], "--b", alpha=0.5)
plt.plot(gn_sub_x[3], [f_erg(i, n) for i in gn_sub_x[3]], "--k", alpha=0.5)

plt.xlabel("Log degree")
plt.ylabel("Log frequency")
plt.legend(loc="upper right")
plt.savefig("img/sub.png", bbox_inches="tight", dpi=150)
  
#### Linear regime ####

#--- Plot 1 ---#
  
# Parameters
n = 10000
reps = 4
thresh_df = 1.5
thresh_d = 2
ms = 5
C_vals = [1/4, 1/2, 1, 2, 4]

# Experiments
gn_al = [[gn_gen_2(n, C) for i in range(reps)] for C in C_vals]
gn_al_df = [[np.asarray(nx.degree_histogram(G)) for G in G_list] for G_list in gn_al]
gn_al_max_len = [max([len(df) for df in df_list]) for df_list in gn_al_df]
gn_al_df = [[np.pad(df, (0, gn_al_max_len[i] - len(df)), mode="constant") for df in gn_al_df[i]] for i in range(len(gn_al))]
gn_al_df = [np.mean(df_list, axis=0) for df_list in gn_al_df]
gn_al_d = [range(len(df)) for df in gn_al_df]
gn_al_inds = [[j for j in gn_al_d[i] if gn_al_df[i][j] >= thresh_df and gn_al_d[i][j] >= thresh_d] for i in range(len(gn_al))]
gn_al_x = [[math.log(gn_al_d[i][j]) for j in gn_al_inds[i]] for i in range(len(gn_al))]
gn_al_y = [[math.log(gn_al_df[i][j]) for j in gn_al_inds[i]] for i in range(len(gn_al))]
gn_al_coef = [np.polyfit(gn_al_x[i], gn_al_y[i], 1) for i in range(len(gn_al))]
gn_al_poly = [np.poly1d(coef) for coef in gn_al_coef]

# Scatter plot
fig, ax = plt.subplots()
ax.set_ylim([0, 9])

plt.plot(gn_al_x[0], gn_al_y[0], "ok", label=f"C = {C_vals[0]} ({round(-gn_al_coef[0][0], 2)})", ms=ms)
plt.plot(gn_al_x[1], gn_al_y[1], "ob", label=f"C = {C_vals[1]} ({round(-gn_al_coef[1][0], 2)})", ms=ms)
plt.plot(gn_al_x[2], gn_al_y[2], "og", label=f"C = {C_vals[2]} ({round(-gn_al_coef[2][0], 2)})", ms=ms)
plt.plot(gn_al_x[3], gn_al_y[3], "oy", label=f"C = {C_vals[3]} ({round(-gn_al_coef[3][0], 2)})", ms=ms)
plt.plot(gn_al_x[4], gn_al_y[4], "or", label=f"C = {C_vals[4]} ({round(-gn_al_coef[4][0], 2)})", ms=ms)

plt.plot(gn_al_x[4], gn_al_poly[0](gn_al_x[4]), "--k", alpha=0.5)
plt.plot(gn_al_x[4], gn_al_poly[1](gn_al_x[4]), "--b", alpha=0.5)
plt.plot(gn_al_x[4], gn_al_poly[2](gn_al_x[4]), "--g", alpha=0.5)
plt.plot(gn_al_x[4], gn_al_poly[3](gn_al_x[4]), "--y", alpha=0.5)
plt.plot(gn_al_x[4], gn_al_poly[4](gn_al_x[4]), "--r", alpha=0.5)

plt.xlabel("Log degree")
plt.ylabel("Log frequency")
plt.legend(loc="upper right")
plt.savefig("img/al_1.png", bbox_inches="tight", dpi=150)

#--- Plot 2 ---#

# Parameters
n_vals = [1000, 5000, 10000, 20000]
reps = 4
thresh_df = 1.5
thresh_d = 2
ms = 5
C_vals = [1/4, 1/2, 1, 2, 4]

gn_al_gamma = [[] for n in n_vals]

for i in range(len(n_vals)):
    n = n_vals[i]
    
    # Experiments
    gn_al = [[gn_gen_2(n, C) for i in range(reps)] for C in C_vals]
    gn_al_df = [[np.asarray(nx.degree_histogram(G)) for G in G_list] for G_list in gn_al]
    gn_al_max_len = [max([len(df) for df in df_list]) for df_list in gn_al_df]
    gn_al_df = [[np.pad(df, (0, gn_al_max_len[i] - len(df)), mode="constant") for df in gn_al_df[i]] for i in range(len(gn_al))]
    gn_al_df = [np.mean(df_list, axis=0) for df_list in gn_al_df]
    gn_al_d = [range(len(df)) for df in gn_al_df]
    gn_al_inds = [[j for j in gn_al_d[i] if gn_al_df[i][j] >= thresh_df and gn_al_d[i][j] >= thresh_d] for i in range(len(gn_al))]
    gn_al_x = [[math.log(gn_al_d[i][j]) for j in gn_al_inds[i]] for i in range(len(gn_al))]
    gn_al_y = [[math.log(gn_al_df[i][j]) for j in gn_al_inds[i]] for i in range(len(gn_al))]
    gn_al_coef = [np.polyfit(gn_al_x[i], gn_al_y[i], 1) for i in range(len(gn_al))]
    
    # Extracting coefficients
    gn_al_gamma[i] = [-coef[0] for coef in gn_al_coef]
    
# Line plot
fig, ax = plt.subplots()

plt.plot(C_vals, [0.5 * (3 + math.sqrt(1 + 8/C)) for C in C_vals], "-or", label="n = ∞", ms=ms)
plt.plot(C_vals, gn_al_gamma[3], "-oy", label=f"n = {n_vals[3]}", ms=ms)
plt.plot(C_vals, gn_al_gamma[2], "-og", label=f"n = {n_vals[2]}", ms=ms)
plt.plot(C_vals, gn_al_gamma[1], "-ob", label=f"n = {n_vals[1]}", ms=ms)
plt.plot(C_vals, gn_al_gamma[0], "-ok", label=f"n = {n_vals[0]}", ms=ms)

plt.xlabel("C")
plt.ylabel("(Estimated) gamma")
plt.legend(loc="upper right")
plt.savefig("img/al_2.png", bbox_inches="tight", dpi=150)

#### Super-linear regime ####

#--- Plot 1 ---#

# Parameters
n = 10000
reps = 4
thresh_df = 0.5
thresh_d = 1
ms = 5
alpha_vals_sup = [1, 1.3, 1.6, 1.9, 2.2]

# Experiments
gn_sup = [[gn_gen_1(n, alpha) for i in range(reps)] for alpha in alpha_vals_sup]
gn_sup_df = [[np.asarray(nx.degree_histogram(G)) for G in G_list] for G_list in gn_sup]
gn_sup_max_len = [max([len(df) for df in df_list]) for df_list in gn_sup_df]
gn_sup_df = [[np.pad(df, (0, gn_sup_max_len[i] - len(df)), mode="constant") for df in gn_sup_df[i]] for i in range(len(gn_sup))]
gn_sup_df = [np.mean(df_list, axis=0) for df_list in gn_sup_df]
gn_sup_d = [range(len(df)) for df in gn_sup_df]
gn_sup_inds = [[j for j in gn_sup_d[i] if gn_sup_df[i][j] >= thresh_df and gn_sup_d[i][j] >= thresh_d] for i in range(len(gn_sup))]
gn_sup_x = [[math.log(gn_sup_d[i][j]) for j in gn_sup_inds[i]] for i in range(len(gn_sup))]
gn_sup_y = [[math.log(gn_sup_df[i][j]) for j in gn_sup_inds[i]] for i in range(len(gn_sup))]

# Scatter plot
fig, ax = plt.subplots()

plt.plot(gn_sup_x[0], gn_sup_y[0], "ok", ms=ms, label=f"α = {alpha_vals_sup[0]} ({round(gn_sup_df[0][1]/n, 3)})")
plt.plot(gn_sup_x[1], gn_sup_y[1], "ob", ms=ms, label=f"α = {alpha_vals_sup[1]} ({round(gn_sup_df[1][1]/n, 3)})")
plt.plot(gn_sup_x[2], gn_sup_y[2], "og", ms=ms, label=f"α = {alpha_vals_sup[2]} ({round(gn_sup_df[2][1]/n, 3)})")
plt.plot(gn_sup_x[3], gn_sup_y[3], "oy", ms=ms, label=f"α = {alpha_vals_sup[3]} ({round(gn_sup_df[3][1]/n, 3)})")
plt.plot(gn_sup_x[4], gn_sup_y[4], "or", ms=ms, label=f"α = {alpha_vals_sup[4]} ({round(gn_sup_df[4][1]/n, 3)})")

plt.xlabel("Log degree")
plt.ylabel("Log frequency")
plt.legend(loc="upper right")
plt.savefig("img/sup_1.png", bbox_inches="tight", dpi=150)

#--- Plot 2 ---#

# Parameters
n_vals = [1000, 5000, 10000, 20000]
reps = 4
thresh_df = 0.5
thresh_d = 1
ms = 5
alpha_vals_sup = [1, 1.3, 1.6, 1.9, 2.2]

gn_sup_prop = [[] for n in n_vals]

for i in range(len(n_vals)):
    n = n_vals[i]
    
    # Experiments
    gn_sup = [[gn_gen_1(n, alpha) for i in range(reps)] for alpha in alpha_vals_sup]
    gn_sup_df = [[np.asarray(nx.degree_histogram(G)) for G in G_list] for G_list in gn_sup]
    gn_sup_max_len = [max([len(df) for df in df_list]) for df_list in gn_sup_df]
    gn_sup_df = [[np.pad(df, (0, gn_sup_max_len[i] - len(df)), mode="constant") for df in gn_sup_df[i]] for i in range(len(gn_sup))]
    gn_sup_df = [np.mean(df_list, axis=0) for df_list in gn_sup_df]
    gn_sup_d = [range(len(df)) for df in gn_sup_df]
    
    # Extracting proportions
    gn_sup_prop[i] = [gn_sup_df[i][1]/n for i in range(len(alpha_vals_sup))]
    
# Line plot
fig, ax = plt.subplots()

plt.plot(alpha_vals_sup, [1 for i in alpha_vals_sup], "-or", label="n = ∞", ms=ms)
plt.plot(alpha_vals_sup, gn_sup_prop[3], "-oy", label=f"n = {n_vals[3]}", ms=ms)
plt.plot(alpha_vals_sup, gn_sup_prop[2], "-og", label=f"n = {n_vals[2]}", ms=ms)
plt.plot(alpha_vals_sup, gn_sup_prop[1], "-ob", label=f"n = {n_vals[1]}", ms=ms)
plt.plot(alpha_vals_sup, gn_sup_prop[0], "-ok", label=f"n = {n_vals[0]}", ms=ms)

plt.xlabel("α")
plt.ylabel("Proportion of 1-degree nodes")
plt.legend(loc="lower right")
plt.savefig("img/sup_2.png", bbox_inches="tight", dpi=150)
