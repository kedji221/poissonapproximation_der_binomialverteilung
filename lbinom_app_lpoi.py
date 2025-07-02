import streamlit as st
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

# Binomial-PMF
def binomial_pmf(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# Poisson-PMF
def poisson_pmf(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

# Listen
def pmf_list(n, p):
    return [binomial_pmf(n, i, p) for i in range(n + 1)]

def poisson_pmf_list(lam, n):
    return [poisson.pmf(i, lam) for i in range(n + 1)]

def cdf_leq_list(vals):
    return [sum(vals[:i+1]) for i in range(len(vals))]

def cdf_geq_list(vals):
    return [sum(vals[i:]) for i in range(len(vals))]

# UI
st.title("🎯 Binomialverteilung & Poisson-Approximation")

# --- Explizite Verteilungsbezeichnungen mit dynamischen Parametern ---
n = st.sidebar.slider("Anzahl der Versuche (n)", 1, 200, 10)
p = st.sidebar.slider("Erfolgswahrscheinlichkeit (p)", 0.0, 1.0, 0.5, step=0.01)
k = st.sidebar.slider("Anzahl der gewünschten Erfolge (k)", 0, n, 3)
#n = st.slider("Anzahl der Versuche (n)", 1, 200, 10)
#p = st.slider("Erfolgswahrscheinlichkeit (p)", 0.0, 1.0, 0.5, step=0.01)
#k = st.slider("Anzahl der gewünschten Erfolge (k)", 0, n, 3)
lam = n * p

# Dynamische Formeln
st.latex(rf"X \sim \mathrm{{Bin}}(n={n},\, p={p:.2f})")
st.latex(rf"Y \sim \mathrm{{Poisson}}(\lambda = n \cdot p = {n} \cdot {p:.2f} = {lam:.2f})")

# Berechnung
pmf_bin = pmf_list(n, p)
pmf_poi = poisson_pmf_list(lam, n)

cdf_leq_bin = cdf_leq_list(pmf_bin)
cdf_leq_poi = cdf_leq_list(pmf_poi)
cdf_geq_bin = cdf_geq_list(pmf_bin)
cdf_geq_poi = cdf_geq_list(pmf_poi)

pmf_k_bin = pmf_bin[k]
pmf_k_poi = pmf_poi[k]
cdf_k_leq_bin = cdf_leq_bin[k]
cdf_k_leq_poi = cdf_leq_poi[k]
cdf_k_geq_bin = cdf_geq_bin[k]
cdf_k_geq_poi = cdf_geq_poi[k]

# -------------------- Diagramm 1: P(X = k) --------------------
fig1, ax1 = plt.subplots()
bars_bin = ax1.bar(np.arange(n + 1) - 0.15, pmf_bin, width=0.3, color='skyblue', label="Binomial")
bars_poi = ax1.bar(np.arange(n + 1) + 0.15, pmf_poi, width=0.3, color='orange', alpha=0.7, label="Poisson")
bars_bin[k].set_color('green')
bars_poi[k].set_color('red')
ax1.set_title("1️⃣ P(X = k): Genau k Erfolge")
ax1.set_xlabel("k")
ax1.set_ylabel("P(X = k)")
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig1)

st.latex(r"P(X = k) = \binom{n}{k} \cdot p^k \cdot (1 - p)^{n - k} \approx \frac{\lambda^k}{k!} e^{-\lambda}")
st.latex(fr"P_{{\text{{Bin}}}}(X = {k}) = {pmf_k_bin:.4f} \qquad P_{{\text{{Pois}}}}(X = {k}) = {pmf_k_poi:.4f}")
st.markdown(f"**Abweichung:** |Binomial - Poisson| = `{abs(pmf_k_bin - pmf_k_poi):.5f}`")

# -------------------- Diagramm 2 + 4: P(X ≤ k) & kumulative Kurve --------------------
col1, col2 = st.columns(2)

with col1:
    fig2, ax2 = plt.subplots()
    ax2.bar(np.arange(n + 1), pmf_bin, color='lightgray', label="Binomial")
    ax2.bar(np.arange(n + 1), pmf_poi, alpha=0.3, color='orange', label="Poisson")
    for i in range(k + 1):
        ax2.bar(i, pmf_bin[i], color='green')
        ax2.bar(i, pmf_poi[i], color='red', alpha=0.5)
    ax2.set_title("2️⃣ P(X ≤ k): Höchstens k Erfolge")
    ax2.set_xlabel("k")
    ax2.set_ylabel("P(X = k)")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend()
    st.pyplot(fig2)

with col2:
    fig4, ax4 = plt.subplots()
    ax4.plot(range(n + 1), cdf_leq_bin, marker='o', color='green', label="Binomial")
    ax4.plot(range(n + 1), cdf_leq_poi, marker='s', color='red', label="Poisson")
    ax4.set_title("4️⃣ Kumulative Verteilung: P(X ≤ x)")
    ax4.set_xlabel("x")
    ax4.set_ylabel("P(X ≤ x)")
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend()
    st.pyplot(fig4)

st.latex(r"P(X \leq k) = \sum_{i = 0}^{k} \binom{n}{i} p^i (1 - p)^{n - i} \approx \sum_{i = 0}^{k} \frac{\lambda^i}{i!} e^{-\lambda}")
st.latex(fr"P_{{\text{{Bin}}}}(X \leq {k}) = {cdf_k_leq_bin:.4f} \qquad P_{{\text{{Pois}}}}(X \leq {k}) = {cdf_k_leq_poi:.4f}")
st.markdown(f"**Abweichung:** |Binomial - Poisson| = `{abs(cdf_k_leq_bin - cdf_k_leq_poi):.5f}`")

# -------------------- Diagramm 3 + 5: P(X ≥ k) & komplementäre Kurve --------------------
col3, col4 = st.columns(2)

with col3:
    fig3, ax3 = plt.subplots()
    ax3.bar(np.arange(n + 1), pmf_bin, color='lightgray', label="Binomial")
    ax3.bar(np.arange(n + 1), pmf_poi, alpha=0.3, color='orange', label="Poisson")
    for i in range(k, n + 1):
        ax3.bar(i, pmf_bin[i], color='green')
        ax3.bar(i, pmf_poi[i], color='red', alpha=0.5)
    ax3.set_title("3️⃣ P(X ≥ k): Mindestens k Erfolge")
    ax3.set_xlabel("k")
    ax3.set_ylabel("P(X = k)")
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.legend()
    st.pyplot(fig3)

with col4:
    fig5, ax5 = plt.subplots()
    ax5.plot(range(n + 1), cdf_geq_bin, marker='o', color='green', label="Binomial")
    ax5.plot(range(n + 1), cdf_geq_poi, marker='s', color='red', label="Poisson")
    ax5.set_title("5️⃣ Komplementäre Kumulative Verteilung: P(X ≥ x)")
    ax5.set_xlabel("x")
    ax5.set_ylabel("P(X ≥ x)")
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.legend()
    st.pyplot(fig5)

st.latex(r"P(X \geq k) = \sum_{i = k}^{n} \binom{n}{i} p^i (1 - p)^{n - i} \approx \sum_{i = k}^{n} \frac{\lambda^i}{i!} e^{-\lambda}")
st.latex(fr"P_{{\text{{Bin}}}}(X \geq {k}) = {cdf_k_geq_bin:.4f} \qquad P_{{\text{{Pois}}}}(X \geq {k}) = {cdf_k_geq_poi:.4f}")
st.markdown(f"**Abweichung:** |Binomial - Poisson| = `{abs(cdf_k_geq_bin - cdf_k_geq_poi):.5f}`")
