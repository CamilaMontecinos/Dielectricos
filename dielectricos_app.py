# app.py
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import streamlit as st

st.set_page_config(page_title="Esfera en dieléctrico", layout="wide")

# ----------------------------
# Parámetros fijos (como tu script original)
# ----------------------------
a = 1.0
Eo = 1.0
Emax = 3.0
NR = 200          # resolución r
NZ = 200          # resolución z
SPAN = 3.0        # extensión +/- 3
STEP_QUIVER = 12  # densidad flechas

@st.cache_data(show_spinner=False)
def precompute_mesh(nr=NR, nz=NZ, span=SPAN):
    r = np.linspace(-span, span, nr)
    z = np.linspace(-span, span, nz)
    R, Z = np.meshgrid(r, z, indexing='xy')
    Rnorm = np.hypot(R, Z)
    return r, z, R, Z, Rnorm

def huefunc_array(x):
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
# Controles (solo los 3 solicitados)
# ----------------------------
st.sidebar.title("Controles")
eps_r = st.sidebar.slider("ε_medio", min_value=1, max_value=100, value=1, step=1)

eps_rsp_val = st.sidebar.slider("ε_esfera", min_value=1, max_value=10, value=1, step=1)
conductor = st.sidebar.checkbox("ε_esfera → ∞ (esfera conductora)", value=False)
if conductor:
    eps_rsp = np.inf
    st.sidebar.caption("Usando ε_esfera = ∞ (se ignora el valor del deslizador).")
else:
    eps_rsp = float(eps_rsp_val)

# ----------------------------
# Cálculo
# ----------------------------
_, _, R, Z, Rnorm = precompute_mesh()
Er, Ez, Emag, img = compute_fields(R, Z, Rnorm, eps_r, eps_rsp, a, Eo, Emax)

# ----------------------------
# Figura única: gráfico + leyenda juntos (pegados)
# ----------------------------
fig, (ax, ax_lg) = plt.subplots(
    1, 2,
    figsize=(6.0, 4.2),  # tamaño físico de la figura (se escalará abajo con width)
    gridspec_kw={'width_ratios': [5, 0.55]}
)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.9, wspace=0.18)

# --- gráfico principal ---
ax.imshow(img, extent=[-SPAN, SPAN, -SPAN, SPAN], origin='lower', aspect='equal')
circle_patch = Circle((0, 0), a, edgecolor='k', fill=False, lw=2)
ax.add_patch(circle_patch)

skip = (slice(None, None, STEP_QUIVER), slice(None, None, STEP_QUIVER))
Er_u = np.zeros_like(Er); Ez_u = np.zeros_like(Ez)
mask = Emag > 0
Er_u[mask] = Er[mask] / Emag[mask]
Ez_u[mask] = Ez[mask] / Emag[mask]
ax.quiver(R[skip], Z[skip], Er_u[skip], Ez_u[skip], color='k')

ax.set_xlabel(r'$r/a$')
ax.set_ylabel(r'$z/a$')
ax.set_title("Campo eléctrico – dieléctrico")
ax.set_xlim(-SPAN, SPAN)
ax.set_ylim(-SPAN, SPAN)
ax.set_aspect('equal', adjustable='box')

# --- leyenda ---
y_legend = np.linspace(0, 3, 300)
ratio_legend = y_legend / 3
colors_legend = huefunc_array(ratio_legend)
legend_img = np.tile(colors_legend[:, None, :], (1, 1, 1))

ax_lg.imshow(legend_img, extent=[0, 1, 0, 3], origin='lower', aspect='auto')
ax_lg.set_xticks([])
ax_lg.set_yticks([0, 1, 2, 3])
ax_lg.set_ylabel(r'$|\!\,E\,\!|/E_0$')
ax_lg.set_title("Leyenda", pad=10)

# Exportar como imagen y centrar en Streamlit con ancho controlado
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
st.image(buf, width=560)  # ← ajusta este ancho (px) para ver más chico o más grande

# ----------------------------
# Caption

st.markdown(
    "<p style='text-align: center; font-size:20px; color:gray;'>© Domenico Sapone, Camila Montecinos</p>",
    unsafe_allow_html=True
)






