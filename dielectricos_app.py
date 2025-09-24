# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:52:21 2025

@author: camil
"""

# app.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import streamlit as st

# ----------------------------
# Utilidades y configuración
# ----------------------------
st.set_page_config(page_title="Esfera en dieléctrico", layout="wide")

# Parámetros base (mantenidos de tu script)
a = 1.0
Eo = 1.0
Emax = 3.0

@st.cache_data(show_spinner=False)
def precompute_mesh(nr=200, nz=200, span=3.0):
    r = np.linspace(-span, span, nr)
    z = np.linspace(-span, span, nz)
    R, Z = np.meshgrid(r, z, indexing='xy')
    Rnorm = np.hypot(R, Z)
    return r, z, R, Z, Rnorm

def huefunc_array(x):
    # x en [0, 1] aprox; fuera de rango se trunca en 0..1 con np.where
    H = np.where(x > 1, 0.0, np.where(x < 0, 0.75, 0.75 * (1 - x)))
    S = np.ones_like(H)
    V = np.ones_like(H)
    HSV = np.stack((H, S, V), axis=-1)
    return mcolors.hsv_to_rgb(HSV)

def compute_fields(R, Z, Rnorm, eps_r, eps_rsp, a, Eo, Emax):
    Er = np.zeros_like(R)
    Ez = np.zeros_like(Z)

    inside = Rnorm < a
    outside = ~inside

    if not np.isinf(eps_rsp):
        Ez[inside] = 3 * eps_r * Eo / (2 * eps_r + eps_rsp)

    if np.isinf(eps_rsp):
        coef = a**3 * Eo
        # Evitar divisiones por 0 en el centro: en outside ya no hay 0
        Er[outside] = 3 * coef * R[outside] * Z[outside] / Rnorm[outside]**5
        Ez[outside] = Eo + coef * (2 * Z[outside]**2 - R[outside]**2) / Rnorm[outside]**5
    else:
        coef = (eps_r - eps_rsp) * a**3 * Eo / (2 * eps_r + eps_rsp)
        Er[outside] = -3 * coef * R[outside] * Z[outside] / Rnorm[outside]**5
        Ez[outside] = Eo - coef * (2 * Z[outside]**2 - R[outside]**2) / Rnorm[outside]**5

    Emag = np.hypot(Er, Ez)
    ratio = Emag / Emax
    img = huefunc_array(ratio)
    return Er, Ez, Emag, img

# ----------------------------
# UI (sidebar)
# ----------------------------
st.sidebar.title("Controles")
eps_r = st.sidebar.slider("ε_medio", min_value=1, max_value=100, value=1, step=1)
conductor = st.sidebar.checkbox("ε_esfera → ∞ (esfera conductora)", value=False)
eps_rsp = np.inf if conductor else st.sidebar.slider("ε_esfera", min_value=1, max_value=100, value=1, step=1)
grid = st.sidebar.select_slider("Resolución de malla", options=[100, 150, 200, 250, 300], value=200)
span = st.sidebar.select_slider("Extensión (±a)", options=[2.0, 2.5, 3.0, 3.5, 4.0], value=3.0)
step_quiver = st.sidebar.select_slider("Densidad flechas", options=[8, 10, 12, 16, 20, 24], value=12)

# ----------------------------
# Cálculo
# ----------------------------
r, z, R, Z, Rnorm = precompute_mesh(nr=grid, nz=grid, span=span)
Er, Ez, Emag, img = compute_fields(R, Z, Rnorm, eps_r, eps_rsp, a, Eo, Emax)

# ----------------------------
# Layout principal
# ----------------------------
col_plot, col_legend = st.columns([5, 1], vertical_alignment="center")

with col_plot:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        img,
        extent=[-span, span, -span, span],
        origin='lower',
        aspect='equal'
    )

    # Esfera
    circle_patch = Circle((0, 0), a, edgecolor='k', fill=False, lw=2)
    ax.add_patch(circle_patch)

    # Flechas normalizadas (unidad)
    skip = (slice(None, None, step_quiver), slice(None, None, step_quiver))
    Er_u = np.zeros_like(Er)
    Ez_u = np.zeros_like(Ez)
    mask = Emag > 0
    Er_u[mask] = Er[mask] / Emag[mask]
    Ez_u[mask] = Ez[mask] / Emag[mask]
    ax.quiver(R[skip], Z[skip], Er_u[skip], Ez_u[skip], color='k', scale=30)

    ax.set_xlabel(r'$r/a$')
    ax.set_ylabel(r'$z/a$')
    ax.set_title("Campo eléctrico – dieléctrico")
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect('equal', adjustable='box')
    st.pyplot(fig, clear_figure=True)

with col_legend:
    # Leyenda vertical |E|/E0 en [0, 3]
    y_legend = np.linspace(0, 3, 300)
    ratio_legend = y_legend / 3
    colors_legend = huefunc_array(ratio_legend)
    legend_img = np.tile(colors_legend[:, None, :], (1, 1, 1))

    fig_lg, ax_lg = plt.subplots(figsize=(1.2, 6))
    ax_lg.imshow(legend_img, extent=[0, 1, 0, 3], origin='lower', aspect='auto')
    ax_lg.set_xticks([])
    ax_lg.set_yticks([0, 1, 2, 3])
    ax_lg.set_ylabel(r'$|\!\,E\,\!|/E_0$')
    ax_lg.set_title("Leyenda", pad=10)
    st.pyplot(fig_lg, clear_figure=True)

st.caption("© Domenico Sapone, Camila Montecinos")
