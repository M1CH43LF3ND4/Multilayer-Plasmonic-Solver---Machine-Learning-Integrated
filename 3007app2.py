import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ==========================================
# 0. WEB PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Fresnel TMM Simulation", layout="wide")
st.title("Multilayer Fresnel Simulation (TMM)")

# ==========================================
# 1. UI: SIDEBAR FOR INPUT PARAMETERS
# ==========================================
st.sidebar.header("⚙️ Input Parameters")
pol = st.sidebar.radio("Polarization Mode", ('P-Pol (TM)', 'S-Pol (TE)'))

# --- Material Mode Settings ---
st.sidebar.markdown("---")
mode_material = st.sidebar.radio(
    "Material System", 
    ("1 Interface (e.g., Air-Glass)", "Multilayer")
)

n_vals = []
k_vals = []
d_vals = []

if mode_material == "1 Interface (e.g., Air-Glass)":
    st.sidebar.markdown("**Medium 1 (Incident)**")
    col1_n, col1_k = st.sidebar.columns(2)
    with col1_n:
        n1 = st.number_input("Re(n)", value=1.0, key="n_bulk_1")
    with col1_k:
        k1 = st.number_input("Im(k)", value=0.0, key="k_bulk_1")
        
    st.sidebar.markdown("**Medium 2 (Transmitted)**")
    col2_n, col2_k = st.sidebar.columns(2)
    with col2_n:
        n2 = st.number_input("Re(n)", value=1.5, key="n_bulk_2")
    with col2_k:
        k2 = st.number_input("Im(k)", value=0.0, key="k_bulk_2")
        
    # Insert inputs into lists
    n_vals = [n1, n2]
    k_vals = [k1, k2]
    # For the case of 1 interface, medium thickness is considered infinite / 0 for TMM calculation
    d_vals = [0.0, 0.0] 

else:
    # Multilayer Mode
    num_layers = st.sidebar.number_input("Number of Layers (Including Incident & Substrate Media)", min_value=3, max_value=10, value=3, step=1)
    
    st.sidebar.markdown("---")
    for i in range(num_layers):
        # Information for each layer
        if i == 0:
            label_layer = f"Layer {i+1} (Incident)"
        elif i == num_layers - 1:
            label_layer = f"Layer {i+1} (Substrate)"
        else:
            label_layer = f"Layer {i+1}"
            
        st.sidebar.markdown(f"**{label_layer}**")
        
        # Divide inputs into 3 columns
        col_n, col_k, col_d = st.sidebar.columns(3)
        with col_n:
            n_val = st.number_input(f"Re(n)", value=1.0 if i==0 else 1.45, key=f"n_{i}")
        with col_k:
            k_val = st.number_input(f"Im(k)", value=0.0, key=f"k_{i}")
        with col_d:
            if i == 0 or i == num_layers - 1:
                d_val = 0.0
                st.text_input(f"d (nm)", value="∞", disabled=True, key=f"d_{i}")
            else:
                d_val = st.number_input(f"d (nm)", value=300.0, min_value=0.0, key=f"d_{i}")
        
        n_vals.append(n_val)
        k_vals.append(k_val)
        d_vals.append(d_val)

st.sidebar.markdown("---")
# Wave Parameters
lam = st.sidebar.slider("Wavelength (nm)", 300.0, 1000.0, 500.0)
th_choice = st.sidebar.slider("Incident Angle (°)", 0.0, 89.9, 0.0)
E0 = st.sidebar.number_input("Amplitude E0 (V/m)", value=100.0)

# Convert lists back to Numpy Arrays
n_vals = np.array(n_vals)
k_vals = np.array(k_vals)
d_vals = np.array(d_vals)
n_c = n_vals + 1j * k_vals

# ==========================================
# 2. PHYSICS FUNCTIONS (N-LAYER TMM GENERAL)
# ==========================================
def get_max_snapshot(comp_arr):
    if np.max(np.abs(comp_arr)) < 1e-12: 
        return np.zeros_like(comp_arr, dtype=float)
    flat_arr = comp_arr.ravel()
    peak_idx = np.argmax(np.abs(flat_arr))
    opt_phase = np.angle(flat_arr[peak_idx])
    return np.real(comp_arr * np.exp(-1j * opt_phase))

def calc_tmm_global(pol, n_c, d_nm, th_arr_rad, lam):
    num_layers = len(n_c)
    kx = n_c[0] * np.sin(th_arr_rad).astype(complex)
    
    cos_th = np.zeros((num_layers, len(th_arr_rad)), dtype=complex)
    kz = np.zeros((num_layers, len(th_arr_rad)), dtype=complex)
    for j in range(num_layers):
        cos_th[j] = np.sqrt(1 - (kx / n_c[j])**2)
        kz[j] = (2 * np.pi / lam) * n_c[j] * cos_th[j]
        
    M11, M12 = np.ones_like(th_arr_rad, dtype=complex), np.zeros_like(th_arr_rad, dtype=complex)
    M21, M22 = np.zeros_like(th_arr_rad, dtype=complex), np.ones_like(th_arr_rad, dtype=complex)
    
    for j in range(num_layers - 1):
        if pol == 'S-Pol (TE)':
            rj = (n_c[j]*cos_th[j] - n_c[j+1]*cos_th[j+1]) / (n_c[j]*cos_th[j] + n_c[j+1]*cos_th[j+1])
            tj = (2 * n_c[j]*cos_th[j]) / (n_c[j]*cos_th[j] + n_c[j+1]*cos_th[j+1])
        else:
            rj = (n_c[j+1]*cos_th[j] - n_c[j]*cos_th[j+1]) / (n_c[j+1]*cos_th[j] + n_c[j]*cos_th[j+1])
            tj = (2 * n_c[j]*cos_th[j]) / (n_c[j+1]*cos_th[j] + n_c[j]*cos_th[j+1])
            
        with np.errstate(divide='ignore', invalid='ignore'):
            nM11 = (M11 + M12 * rj) / tj
            nM12 = (M11 * rj + M12) / tj
            nM21 = (M21 + M22 * rj) / tj
            nM22 = (M21 * rj + M22) / tj
        
        if j < num_layers - 2:
            delta = kz[j+1] * d_nm[j+1]
            P11, P22 = np.exp(-1j * delta), np.exp(1j * delta)
            M11, M12 = nM11 * P11, nM12 * P22
            M21, M22 = nM21 * P11, nM22 * P22
        else:
            M11, M12, M21, M22 = nM11, nM12, nM21, nM22
            
    with np.errstate(divide='ignore', invalid='ignore'):
        r_tot = M21 / M11
        t_tot = 1 / M11
        R_pow = np.abs(r_tot)**2
        factor = np.real(n_c[-1] * cos_th[-1]) / np.real(n_c[0] * cos_th[0])
        T_pow = np.abs(t_tot)**2 * factor
        
    R_pow = np.nan_to_num(R_pow, nan=1.0)
    T_pow = np.nan_to_num(T_pow, nan=0.0)
    A_pow = np.clip(1.0 - R_pow - T_pow, 0, 1)
    
    return R_pow, T_pow, A_pow, r_tot, t_tot

# ==========================================
# 3. 1D TMM CALCULATION (FRACTIONS)
# ==========================================
# --- A. Setup Arrays ---
theta_array_deg = np.linspace(0, 89.9, 500)
theta_array_rad = np.radians(theta_array_deg)

lam_array_nm = np.linspace(300.0, 1000.0, 500)
th_fixed_rad = np.array([np.radians(th_choice)]) 

# --- B. Calc 1: vs Angle (Fixed Wavelength) ---
R_arr, T_arr, A_arr, r_arr, t_arr = calc_tmm_global(pol, n_c, d_vals, theta_array_rad, lam)

# --- C. Calc 2: vs Wavelength (Fixed Angle) ---
R_lam_list, T_lam_list, A_lam_list = [], [], []

for wl in lam_array_nm:
    R_val, T_val, A_val, _, _ = calc_tmm_global(pol, n_c, d_vals, th_fixed_rad, wl)
    R_lam_list.append(R_val[0])
    T_lam_list.append(T_val[0])
    A_lam_list.append(A_val[0])

R_lam_arr = np.array(R_lam_list)
T_lam_arr = np.array(T_lam_list)
A_lam_arr = np.array(A_lam_list)

# --- D. Fundamental Physics Info Text ---
n0 = np.real(n_c[0])
n1 = np.real(n_c[1])
n_sub = np.real(n_c[-1])
th_B_1 = np.degrees(np.arctan(n1 / n0))
th_B_sub = np.degrees(np.arctan(n_sub / n0))
str_angles = f"★ Brewster Angle (Incident -> Layer 1): {th_B_1:.2f}° | Brewster Angle (Substrate): {th_B_sub:.2f}°"
if n0 > n_sub:
    th_C_sub = np.degrees(np.arcsin(n_sub / n0))
    str_angles += f" | Critical Angle: {th_C_sub:.2f}°"

st.info(str_angles)

# --- E. 1D Plots to Streamlit ---
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(theta_array_deg, R_arr, color='blue', lw=2, label='Reflectance ($R$)')
ax1.plot(theta_array_deg, T_arr, color='red', lw=2, label='Transmittance ($T$)')
ax1.plot(theta_array_deg, A_arr, color='green', lw=2, label='Absorptance ($A$)')
ax1.axvline(x=th_choice, color='gray', linestyle=':', label='Current Angle')
ax1.set_title(f'R, T, A Fractions vs Incident Angle ($\lambda$ = {lam} nm)', fontweight='bold')
ax1.set_xlabel('Incident Angle (degrees)')
ax1.set_ylabel('R, T, A Fractions')
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(0, 90)
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(lam_array_nm, R_lam_arr, color='blue', lw=2, label='Reflectance ($R$)')
ax2.plot(lam_array_nm, T_lam_arr, color='red', lw=2, label='Transmittance ($T$)')
ax2.plot(lam_array_nm, A_lam_arr, color='green', lw=2, label='Absorptance ($A$)')
ax2.axvline(x=lam, color='gray', linestyle=':', label='Current $\lambda$')
ax2.set_title(f'R, T, A Fractions vs Wavelength ($\theta$ = {th_choice}°)', fontweight='bold')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('R, T, A Fractions')
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlim(300, 1000)
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# ==========================================
# 4. TMM BACKPROPAGATION 2D (FIELD PROFILES)
# ==========================================
num_layers = len(n_c)
total_thickness = np.sum(d_vals[1:-1]) / 1000.0
z_max = max(2.5, total_thickness + 1.0)
x_um = np.linspace(-1.5, 1.5, 120)
z_um = np.linspace(-1.5, z_max, 180) 
X, Z = np.meshgrid(x_um, z_um)

d_um = d_vals / 1000.0
z_bounds = [0.0]
for i in range(1, num_layers - 1):
    z_bounds.append(z_bounds[-1] + d_um[i])

th_single = np.radians(th_choice)
kx = n_c[0] * np.sin(th_single)
cos_th_s = np.zeros(num_layers, dtype=complex)
kz_s = np.zeros(num_layers, dtype=complex)

for j in range(num_layers):
    cos_th_s[j] = np.sqrt(1 - (kx / n_c[j])**2)
    kz_s[j] = (2 * np.pi / lam) * n_c[j] * cos_th_s[j]
    
r_j, t_j = np.zeros(num_layers-1, dtype=complex), np.zeros(num_layers-1, dtype=complex)
for j in range(num_layers - 1):
    if pol == 'S-Pol (TE)':
        r_j[j] = (n_c[j]*cos_th_s[j] - n_c[j+1]*cos_th_s[j+1]) / (n_c[j]*cos_th_s[j] + n_c[j+1]*cos_th_s[j+1])
        t_j[j] = (2 * n_c[j]*cos_th_s[j]) / (n_c[j]*cos_th_s[j] + n_c[j+1]*cos_th_s[j+1])
    else: 
        r_j[j] = (n_c[j+1]*cos_th_s[j] - n_c[j]*cos_th_s[j+1]) / (n_c[j+1]*cos_th_s[j] + n_c[j]*cos_th_s[j+1])
        t_j[j] = (2 * n_c[j]*cos_th_s[j]) / (n_c[j+1]*cos_th_s[j] + n_c[j]*cos_th_s[j+1])
        
_, _, _, r_single, t_single = calc_tmm_global(pol, n_c, d_vals, np.array([th_single]), lam)
v, w = np.zeros(num_layers, dtype=complex), np.zeros(num_layers, dtype=complex)
v[-1], w[-1] = E0 * t_single[0], 0

for j in range(num_layers - 2, -1, -1):
    with np.errstate(divide='ignore', invalid='ignore'):
        v_end = (v[j+1] + r_j[j] * w[j+1]) / t_j[j]
        w_end = (r_j[j] * v[j+1] + w[j+1]) / t_j[j]
    if j > 0:
        delta = kz_s[j] * d_vals[j]
        v[j] = v_end * np.exp(-1j * delta)
        w[j] = w_end * np.exp(1j * delta)
    else:
        v[j], w[j] = v_end, w_end

Ex = np.zeros_like(X, dtype=complex); Ey = np.zeros_like(X, dtype=complex)
Ez = np.zeros_like(X, dtype=complex); Hy = np.zeros_like(X, dtype=complex)
Z0 = 376.7303
kx_um = (2 * np.pi / (lam / 1000.0)) * n_c[0] * np.sin(th_single)
kz_um = kz_s * 1000.0 

U_prop = np.zeros_like(X, dtype=float); V_prop = np.zeros_like(X, dtype=float)

for j in range(num_layers):
    if j == 0:
        mask = Z < z_bounds[0]
        z_loc = Z[mask] - z_bounds[0]
    elif j == num_layers - 1:
        mask = Z >= z_bounds[-1]
        z_loc = Z[mask] - z_bounds[-1]
    else:
        mask = (Z >= z_bounds[j-1]) & (Z < z_bounds[j])
        z_loc = Z[mask] - z_bounds[j-1]
        
    phase_v = np.exp(1j * kz_um[j] * z_loc)
    phase_w = np.exp(-1j * kz_um[j] * z_loc)
    fasa_x = np.exp(1j * kx_um * X[mask])
    
    if pol == 'P-Pol (TM)':
        Ex[mask] = (v[j] * cos_th_s[j] * phase_v - w[j] * cos_th_s[j] * phase_w) * fasa_x
        st_j = n_c[0] * np.sin(th_single) / n_c[j]
        Ez[mask] = (-v[j] * st_j * phase_v - w[j] * st_j * phase_w) * fasa_x
        Hy[mask] = (n_c[j] / Z0) * (v[j] * phase_v + w[j] * phase_w) * fasa_x
    else:
        Ey[mask] = (v[j] * phase_v + w[j] * phase_w) * fasa_x
        Hy[mask] = (-n_c[j] * cos_th_s[j] / Z0) * (v[j] * phase_v - w[j] * phase_w) * fasa_x
        
    if np.imag(cos_th_s[j]) == 0:
        uj = np.real(n_c[0] * np.sin(th_single) / n_c[j])
        vj = -np.real(cos_th_s[j])
        norm = np.sqrt(uj**2 + vj**2)
    else:
        uj, vj, norm = 1.0, 0.0, 1.0
    U_prop[mask] = uj / norm; V_prop[mask] = vj / norm

Ex_real, Ey_real = get_max_snapshot(Ex), get_max_snapshot(Ey)
Ez_real, Hy_real = get_max_snapshot(Ez), get_max_snapshot(Hy)

vmax_val = max(np.max(np.abs(Ex_real)), np.max(np.abs(Ey_real)), np.max(np.abs(Ez_real)))
if vmax_val == 0: vmax_val = E0 if E0 > 0 else 1e-10 
vmax_h = 0.6 * (E0 / 100.0) if E0 > 0 else 1e-10 

# ==========================================
# 5. PLOTTING 2D FIELDS TO STREAMLIT
# ==========================================
st.markdown("### 2D Electromagnetic Field Profiles")
col3, col4, col5, col6 = st.columns(4)
step = 15 

def setup_2d_plot(ax, title, z_bounds):
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Position X ($\mu$m)')
    ax.set_ylabel('Depth Z ($\mu$m)')
    ax.set_ylim(z_max, -1.5)
    ax.set_xlim(-1.5, 1.5)
    for b_z in z_bounds:
        ax.axhline(b_z, color='white', linestyle='--', lw=1.5, alpha=0.9)

with col3:
    fig_ex, ax_ex = plt.subplots(figsize=(4, 6))
    mesh_ex = ax_ex.pcolormesh(X, Z, Ex_real, cmap='jet', shading='gouraud', vmin=-vmax_val, vmax=vmax_val)
    if pol == 'P-Pol (TM)':
        ax_ex.quiver(X[::step, ::step], Z[::step, ::step], U_prop[::step, ::step], V_prop[::step, ::step], color='red', pivot='mid', scale=20, alpha=0.9)
    setup_2d_plot(ax_ex, 'Field Profile $E_x$', z_bounds)
    fig_ex.colorbar(mesh_ex, ax=ax_ex, fraction=0.046, pad=0.04).set_label('Amplitude E (V/m)')
    st.pyplot(fig_ex)

with col4:
    fig_ey, ax_ey = plt.subplots(figsize=(4, 6))
    mesh_ey = ax_ey.pcolormesh(X, Z, Ey_real, cmap='jet', shading='gouraud', vmin=-vmax_val, vmax=vmax_val)
    setup_2d_plot(ax_ey, 'Field Profile $E_y$', z_bounds)
    fig_ey.colorbar(mesh_ey, ax=ax_ey, fraction=0.046, pad=0.04).set_label('Amplitude E (V/m)')
    st.pyplot(fig_ey)

with col5:
    fig_ez, ax_ez = plt.subplots(figsize=(4, 6))
    mesh_ez = ax_ez.pcolormesh(X, Z, Ez_real, cmap='jet', shading='gouraud', vmin=-vmax_val, vmax=vmax_val)
    if pol == 'P-Pol (TM)':
        ax_ez.quiver(X[::step, ::step], Z[::step, ::step], U_prop[::step, ::step], V_prop[::step, ::step], color='red', pivot='mid', scale=20, alpha=0.9)
    setup_2d_plot(ax_ez, 'Field Profile $E_z$', z_bounds)
    fig_ez.colorbar(mesh_ez, ax=ax_ez, fraction=0.046, pad=0.04).set_label('Amplitude E (V/m)')
    st.pyplot(fig_ez)

with col6:
    fig_h, ax_h = plt.subplots(figsize=(4, 6))
    mesh_h = ax_h.pcolormesh(X, Z, Hy_real, cmap='jet', shading='gouraud', vmin=-vmax_h, vmax=vmax_h)
    title_h = 'Field Profile $H_y$' if pol == 'P-Pol (TM)' else 'Field Profile $H_x$'
    setup_2d_plot(ax_h, title_h, z_bounds)
    fig_h.colorbar(mesh_h, ax=ax_h, fraction=0.046, pad=0.04).set_label('Amplitude H (A/m)')
    st.pyplot(fig_h)

# ==========================================
# 6. DATA CALCULATE AND EXTRACTION BUTTON 
# ==========================================
st.markdown("---")
if st.button("Calculate & Prepare Extraction Data (TM & TE)"):
    R_tm, T_tm, A_tm, _, _ = calc_tmm_global('P-Pol (TM)', n_c, d_vals, theta_array_rad, lam)
    R_te, T_te, A_te, _, _ = calc_tmm_global('S-Pol (TE)', n_c, d_vals, theta_array_rad, lam)
    
    # Remove Er and Et from extraction data
    data_matrix = np.column_stack((
        theta_array_deg, 
        R_tm, T_tm, A_tm, 
        R_te, T_te, A_te
    ))
    
    header_text = 'Angle(deg)\tR_TM\tT_TM\tA_TM\tR_TE\tT_TE\tA_TE'
    
    import io
    csv_buffer = io.BytesIO()
    np.savetxt(csv_buffer, data_matrix, fmt='%.6f', delimiter='\t', header=header_text, comments='')
    
    st.download_button(
        label="📥 Download File Data_Fresnel_TMM_Complete.txt",
        data=csv_buffer.getvalue(),
        file_name="Data_Fresnel_TMM_Complete.txt",
        mime="text/plain"
    )

# ==========================================
# 7. AI INVERSE DESIGN MODULE (Single Layer Dynamic)
# ==========================================
st.markdown("---")
st.markdown("## Inverse Design - Machine Learning (incident-material-substrate)")
st.markdown("Specify your optical system media ($n_{inc}$ and $n_{sub}$) and film material ($n_{film}$). The AI will calculate physical limits and predict the required thickness.")

# Setup Material Inputs
col_mat1, col_mat2, col_mat3, col_mat4 = st.columns(4)
with col_mat1:
    target_lam = st.number_input("Target Wavelength (nm)", min_value=400.0, max_value=800.0, value=500.0, step=10.0)
with col_mat2:
    # Perlebar max_value menjadi 4.0 (atau sesuai batas n_inc di dataset Anda)
    target_n_inc = st.number_input("Incident Medium ($n_{inc}$)", min_value=1.0, max_value=4.0, value=1.00, step=0.05)
with col_mat3:
    # Perlebar max_value menjadi 5.0 (atau sesuai batas n_film di dataset Anda)
    target_n_film = st.number_input("Film Material ($n_{film}$)", min_value=1.0, max_value=5.0, value=1.45, step=0.05)
with col_mat4:
    # Perlebar max_value menjadi 5.0 (atau sesuai batas n_sub di dataset Anda)
    target_n_sub = st.number_input("Substrate Medium ($n_{sub}$)", min_value=1.0, max_value=5.0, value=1.50, step=0.05)
# --- DYNAMIC PHYSICAL BOUNDARY CALCULATION ---
# R at zero thickness (bare substrate interface)
R_bare = ((target_n_inc - target_n_sub) / (target_n_inc + target_n_sub))**2

# R at quarter-wave optical thickness (peak interference)
R_qwave = ((target_n_film**2 - (target_n_inc * target_n_sub)) / (target_n_film**2 + (target_n_inc * target_n_sub)))**2

max_possible_R = max(R_bare, R_qwave)
min_possible_R = min(R_bare, R_qwave)

# Inform User of Physics Boundaries
st.info(f"💡 **Physics Boundaries:** For $n_{{inc}}={target_n_inc:.2f}$, $n_{{film}}={target_n_film:.2f}$, and $n_{{sub}}={target_n_sub:.2f}$, the reflectance $R$ can only oscillate between **{min_possible_R:.4f}** and **{max_possible_R:.4f}**.")

# Dynamic Target Slider
target_R = st.slider(
    "Set Desired Target Reflectance (R)", 
    min_value=float(min_possible_R), 
    max_value=float(max_possible_R), 
    value=float(min_possible_R), 
    step=0.001
)

if st.button("Generate Optimal Thickness"):
    try:
        # Load Dynamic Single Layer Model
        rf_model = joblib.load("rf_tmm_model_single_dynamic.pkl.bz2")
        
        # 5 Input Features: [Wavelength, n_inc, n_film, n_sub, Target_R]
        input_features = np.array([[target_lam, target_n_inc, target_n_film, target_n_sub, target_R]])
        pred_d = rf_model.predict(input_features)[0]
        
        st.success("AI Design Successfully Formulated!")
        st.metric(label="Recommended Film Thickness (d)", value=f"{pred_d:.2f} nm")
        
        # --- PHYSICS VALIDATION & ERROR ANALYSIS ---
        st.markdown("### Physical Validation (TMM Analytical Cross-Check)")
        lam_array = np.linspace(300, 1000, 300)
        n_array_val = np.array([target_n_inc + 0j, target_n_film + 0j, target_n_sub + 0j])
        d_array_val = np.array([0.0, pred_d, 0.0])
        th_val = np.array([0.0]) # Normal incidence
        
        # Calculate actual TMM Reflectance at target wavelength
        R_actual_target, _, _, _, _ = calc_tmm_global('S-Pol (TE)', n_array_val, d_array_val, th_val, target_lam)
        R_actual_val = R_actual_target[0]
        
        abs_error = abs(R_actual_val - target_R)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Analytical TMM Reflectance", f"{R_actual_val:.4f}", delta=f"{R_actual_val - target_R:.4f} (Error)", delta_color="inverse")
        
        if target_R > 1e-6:
            rel_error = (abs_error / target_R) * 100
            col_m2.metric("Relative Error (%)", f"{rel_error:.2f} %")
        else:
            col_m2.metric("Relative Error (%)", "N/A (Target is 0)")
            
        st.markdown("---")

        # --- VALIDATION SPECTRUM PLOT ---
        R_val_list = []
        for wl in lam_array:
            R_val, _, _, _, _ = calc_tmm_global('S-Pol (TE)', n_array_val, d_array_val, th_val, wl)
            R_val_list.append(R_val[0])
            
        fig_ai, ax_ai = plt.subplots(figsize=(10, 4))
        ax_ai.plot(lam_array, R_val_list, color='purple', lw=2, label=f'ML Configuration ($d$={pred_d:.1f}nm)')
        ax_ai.scatter([target_lam], [target_R], color='red', s=100, zorder=5, label='User Target')
        ax_ai.axvline(x=target_lam, color='gray', linestyle='--')
        ax_ai.axhline(y=target_R, color='gray', linestyle='--')
        
        ax_ai.set_title('Validation Spectrum: Machine Learning Configuration vs Target', fontweight='bold')
        ax_ai.set_xlabel('Wavelength (nm)')
        ax_ai.set_ylabel('Reflectance (R)')
        ax_ai.set_xlim(300, 1000)
        ax_ai.legend()
        ax_ai.grid(True, alpha=0.3)
        st.pyplot(fig_ai)
        
    except FileNotFoundError:
        st.error("Model file 'rf_tmm_model_single_dynamic.pkl' not found. Please run Step 1 & 2 scripts first.")
