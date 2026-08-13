import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

# ==========================================
# 0. WEB PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Dispersive TMM & Plasmonics", layout="wide")
st.title("Multilayer Plasmonic Solver")

# ==========================================
# 1. HELPER: DATA PARSER & PHYSICS CONVERTER
# ==========================================
def parse_uploaded_file(uploaded_file):
    """Membaca file data mentah (CSV/TXT) dan menghapus baris teks/header"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        clean_lines = []
        for line in content.split('\n'):
            if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '-'):
                clean_lines.append(line)
        
        clean_content = '\n'.join(clean_lines)
        if ',' in clean_lines[0]:
            data = np.loadtxt(io.StringIO(clean_content), delimiter=',')
        else:
            data = np.loadtxt(io.StringIO(clean_content))
            
        return data[:, 0], data[:, 1], data[:, 2] # Wavelength, Real, Imag
    except Exception as e:
        return None, None, None

def get_nc_array(layer_mats, current_lam, input_type):
    """Menghitung array Indeks Bias Dinamis (n_c) pada panjang gelombang tertentu"""
    nc_list = []
    for mat in layer_mats:
        if mat['type'] == 'constant':
            r_val = mat['r']
            i_val = mat['i']
        else:
            # Interpolasi linier jika data material berupa spektrum dispersif (CSV/TXT)
            r_val = np.interp(current_lam, mat['wl'], mat['r'])
            i_val = np.interp(current_lam, mat['wl'], mat['i'])
        
        # Konversi Permitivitas ke Indeks Bias (jika mode Permitivitas aktif)
        if input_type == "Permittivity (ε1, ε2)":
            eps = r_val + 1j * i_val
            nc = np.sqrt(eps)
        else:
            nc = r_val + 1j * i_val
            
        nc_list.append(nc)
    return np.array(nc_list)

# ==========================================
# 2. UI: SIDEBAR FOR INPUT PARAMETERS
# ==========================================
st.sidebar.header("⚙️ System Parameters")
pol = st.sidebar.radio("Polarization Mode", ('P-Pol (TM)', 'S-Pol (TE)'))

st.sidebar.markdown("---")
# Fitur Baru: Pemilih Mode Input Optik Global
input_type = st.sidebar.radio(
    "Optical Parameter Type", 
    ("Refractive Index (n, k)", "Permittivity (ε1, ε2)")
)
lbl_r = "Re(n)" if input_type == "Refractive Index (n, k)" else "ε1 (Real)"
lbl_i = "Im(k)" if input_type == "Refractive Index (n, k)" else "ε2 (Imag)"

st.sidebar.markdown("---")
mode_material = st.sidebar.radio(
    "Architecture", 
    ("1 Interface (e.g., Air-Glass)", "Multilayer")
)

layer_mats = []

if mode_material == "1 Interface (e.g., Air-Glass)":
    num_layers = 2
else:
    num_layers = st.sidebar.number_input("Number of Layers (Including Incident & Substrate)", min_value=3, max_value=10, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Material Definition")

for i in range(num_layers):
    if i == 0:
        label_layer = f"Layer {i+1} (Incident)" if num_layers > 2 else "Medium 1 (Incident)"
    elif i == num_layers - 1:
        label_layer = f"Layer {i+1} (Substrate)" if num_layers > 2 else "Medium 2 (Transmitted)"
    else:
        label_layer = f"Layer {i+1} (Film)"
        
    st.sidebar.markdown(f"**{label_layer}**")
    
    # Fitur Baru: Pemilih Sumber Material per Lapisan
    mat_source = st.sidebar.selectbox(f"Data Source", ["Constant Value", "Import File (.csv/.txt)"], key=f"src_{i}")
    
    d_val = 0.0
    if mat_source == "Constant Value":
        col_r, col_i, col_d = st.sidebar.columns(3)
        with col_r:
            val_r = st.number_input(lbl_r, value=1.0 if i==0 else 1.45, format="%.4f", key=f"r_{i}")
        with col_i:
            val_i = st.number_input(lbl_i, value=0.0, format="%.4f", key=f"i_{i}")
        with col_d:
            if i == 0 or i == num_layers - 1:
                st.text_input(f"d (nm)", value="∞", disabled=True, key=f"d_{i}")
            else:
                d_val = st.number_input(f"d (nm)", value=50.0, min_value=0.0, key=f"d_{i}")
                
        layer_mats.append({'type': 'constant', 'r': val_r, 'i': val_i, 'd': d_val})
        
    else:
        # Fitur Baru: Pengunggah CSV/TXT
        file = st.sidebar.file_uploader(f"Upload 3-Column Data (WL, {lbl_r}, {lbl_i})", type=['csv', 'txt'], key=f"file_{i}")
        with st.sidebar.columns(3)[2]:
            if i == 0 or i == num_layers - 1:
                st.text_input(f"d (nm)", value="∞", disabled=True, key=f"d_f_{i}")
            else:
                d_val = st.number_input(f"d (nm)", value=50.0, min_value=0.0, key=f"d_f_{i}")
                
        if file is not None:
            wl_data, r_data, i_data = parse_uploaded_file(file)
            if wl_data is not None:
                layer_mats.append({'type': 'dispersive', 'wl': wl_data, 'r': r_data, 'i': i_data, 'd': d_val})
                st.sidebar.success("✅ Data Loaded")
            else:
                st.sidebar.error("❌ Format Error. Need numbers only.")
                layer_mats.append({'type': 'constant', 'r': 1.0, 'i': 0.0, 'd': d_val})
        else:
            layer_mats.append({'type': 'constant', 'r': 1.0, 'i': 0.0, 'd': d_val})

st.sidebar.markdown("---")
# Wave Parameters
lam = st.sidebar.slider("Global Evaluation Wavelength (nm)", 300.0, 1000.0, 500.0)
th_choice = st.sidebar.slider("Incident Angle (°)", 0.0, 89.9, 0.0)
E0 = st.sidebar.number_input("Amplitude E0 (V/m)", value=100.0)

# Array ekstraksi ketebalan untuk core TMM
d_vals = np.array([mat['d'] for mat in layer_mats])

# ==========================================
# 3. PHYSICS FUNCTIONS (TMM)
# ==========================================
def get_max_snapshot(comp_arr):
    if np.max(np.abs(comp_arr)) < 1e-12: 
        return np.zeros_like(comp_arr, dtype=float)
    flat_arr = comp_arr.ravel()
    peak_idx = np.argmax(np.abs(flat_arr))
    opt_phase = np.angle(flat_arr[peak_idx])
    return np.real(comp_arr * np.exp(-1j * opt_phase))

def calc_tmm_global(pol, n_c, d_nm, th_arr_rad, lam_val):
    num_layers = len(n_c)
    kx = n_c[0] * np.sin(th_arr_rad).astype(complex)
    
    cos_th = np.zeros((num_layers, len(th_arr_rad)), dtype=complex)
    kz = np.zeros((num_layers, len(th_arr_rad)), dtype=complex)
    for j in range(num_layers):
        cos_th[j] = np.sqrt(1 - (kx / n_c[j])**2)
        kz[j] = (2 * np.pi / lam_val) * n_c[j] * cos_th[j]
        
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
# 4. 1D TMM CALCULATION & DYNAMIC PLOTTING
# ==========================================
st.markdown("### 1D Analytical Results (R, T, A)")

theta_array_deg = np.linspace(0, 89.9, 500)
theta_array_rad = np.radians(theta_array_deg)
lam_array_nm = np.linspace(300.0, 1000.0, 500)
th_fixed_rad = np.array([np.radians(th_choice)]) 

# Kalkulasi 1: vs Incident Angle (Spektrum dinamis berpusat di lam)
n_c_lam = get_nc_array(layer_mats, lam, input_type)
R_arr, T_arr, A_arr, _, _ = calc_tmm_global(pol, n_c_lam, d_vals, theta_array_rad, lam)

# Kalkulasi 2: vs Wavelength (TMM dieksekusi berulang karena efek dispersi material)
R_lam_list, T_lam_list, A_lam_list = [], [], []
for wl in lam_array_nm:
    n_c_wl = get_nc_array(layer_mats, wl, input_type)
    R_val, T_val, A_val, _, _ = calc_tmm_global(pol, n_c_wl, d_vals, th_fixed_rad, wl)
    R_lam_list.append(R_val[0])
    T_lam_list.append(T_val[0])
    A_lam_list.append(A_val[0])

R_lam_arr = np.array(R_lam_list)
T_lam_arr = np.array(T_lam_list)
A_lam_arr = np.array(A_lam_list)

# Info Brewster & Critical Angle (dievaluasi pada lam)
n0 = np.real(n_c_lam[0])
n1 = np.real(n_c_lam[1])
n_sub = np.real(n_c_lam[-1])
th_B_1 = np.degrees(np.arctan(n1 / n0))
th_B_sub = np.degrees(np.arctan(n_sub / n0))
str_angles = f"★ Evaluation at $\lambda$ = {lam} nm | Brewster Angle (Incident $\\rightarrow$ Layer 1): {th_B_1:.2f}° | Brewster Angle (Substrate): {th_B_sub:.2f}°"
if n0 > n_sub:
    th_C_sub = np.degrees(np.arcsin(n_sub / n0))
    str_angles += f" | Critical Angle: {th_C_sub:.2f}°"

st.info(str_angles)

# Plot Toggle Mode (Gabungan Tampilan Grafik)
plot_mode = st.radio(
    "Select Plot View Mode:", 
    ("Incident Angle θ (Fixed Wavelength)", "Wavelength $\lambda$ (Fixed Incident Angle)"), 
    horizontal=True
)

fig, ax = plt.subplots(figsize=(10, 4))

if plot_mode == "Incident Angle θ (Fixed Wavelength)":
    ax.plot(theta_array_deg, R_arr, color='blue', lw=2, label='Reflectance ($R$)')
    ax.plot(theta_array_deg, T_arr, color='red', lw=2, label='Transmittance ($T$)')
    ax.plot(theta_array_deg, A_arr, color='green', lw=2, label='Absorptance ($A$)')
    ax.axvline(x=th_choice, color='gray', linestyle=':', label='Current Angle')
    ax.set_title(f'R, T, A Fractions vs Incident Angle ($\lambda$ = {lam} nm)', fontweight='bold')
    ax.set_xlabel('Incident Angle (degrees)')
    ax.set_xlim(0, 90)
else:
    ax.plot(lam_array_nm, R_lam_arr, color='blue', lw=2, label='Reflectance ($R$)')
    ax.plot(lam_array_nm, T_lam_arr, color='red', lw=2, label='Transmittance ($T$)')
    ax.plot(lam_array_nm, A_lam_arr, color='green', lw=2, label='Absorptance ($A$)')
    ax.axvline(x=lam, color='gray', linestyle=':', label='Current $\lambda$')
    ax.set_title(f'R, T, A Fractions vs Wavelength ($\\theta$ = {th_choice}°)', fontweight='bold')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim(300, 1000)

ax.set_ylabel('R, T, A Fractions')
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ==========================================
# 5. TMM BACKPROPAGATION 2D (FIELD PROFILES)
# ==========================================
# Evaluasi matriks medan pada kondisi statis (lam dan th_choice)
n_c_2d = get_nc_array(layer_mats, lam, input_type)

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
kx = n_c_2d[0] * np.sin(th_single)
cos_th_s = np.zeros(num_layers, dtype=complex)
kz_s = np.zeros(num_layers, dtype=complex)

for j in range(num_layers):
    cos_th_s[j] = np.sqrt(1 - (kx / n_c_2d[j])**2)
    kz_s[j] = (2 * np.pi / lam) * n_c_2d[j] * cos_th_s[j]
    
r_j, t_j = np.zeros(num_layers-1, dtype=complex), np.zeros(num_layers-1, dtype=complex)
for j in range(num_layers - 1):
    if pol == 'S-Pol (TE)':
        r_j[j] = (n_c_2d[j]*cos_th_s[j] - n_c_2d[j+1]*cos_th_s[j+1]) / (n_c_2d[j]*cos_th_s[j] + n_c_2d[j+1]*cos_th_s[j+1])
        t_j[j] = (2 * n_c_2d[j]*cos_th_s[j]) / (n_c_2d[j]*cos_th_s[j] + n_c_2d[j+1]*cos_th_s[j+1])
    else: 
        r_j[j] = (n_c_2d[j+1]*cos_th_s[j] - n_c_2d[j]*cos_th_s[j+1]) / (n_c_2d[j+1]*cos_th_s[j] + n_c_2d[j]*cos_th_s[j+1])
        t_j[j] = (2 * n_c_2d[j]*cos_th_s[j]) / (n_c_2d[j+1]*cos_th_s[j] + n_c_2d[j]*cos_th_s[j+1])
        
_, _, _, r_single, t_single = calc_tmm_global(pol, n_c_2d, d_vals, np.array([th_single]), lam)
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
kx_um = (2 * np.pi / (lam / 1000.0)) * n_c_2d[0] * np.sin(th_single)
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
    phase_x = np.exp(1j * kx_um * X[mask])
    
    if pol == 'P-Pol (TM)':
        Ex[mask] = (v[j] * cos_th_s[j] * phase_v - w[j] * cos_th_s[j] * phase_w) * phase_x
        st_j = n_c_2d[0] * np.sin(th_single) / n_c_2d[j]
        Ez[mask] = (-v[j] * st_j * phase_v - w[j] * st_j * phase_w) * phase_x
        Hy[mask] = (n_c_2d[j] / Z0) * (v[j] * phase_v + w[j] * phase_w) * phase_x
    else:
        Ey[mask] = (v[j] * phase_v + w[j] * phase_w) * phase_x
        Hy[mask] = (-n_c_2d[j] * cos_th_s[j] / Z0) * (v[j] * phase_v - w[j] * phase_w) * phase_x
        
    if np.imag(cos_th_s[j]) == 0:
        uj = np.real(n_c_2d[0] * np.sin(th_single) / n_c_2d[j])
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

st.markdown("---")
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
    # Karena ekstraksi dilakukan terhadap sudut incident (Angle) dan lam tetap, 
    # evaluasi materi dilakukan sekali di n_c_lam
    R_tm, T_tm, A_tm, _, _ = calc_tmm_global('P-Pol (TM)', n_c_lam, d_vals, theta_array_rad, lam)
    R_te, T_te, A_te, _, _ = calc_tmm_global('S-Pol (TE)', n_c_lam, d_vals, theta_array_rad, lam)
    
    data_matrix = np.column_stack((
        theta_array_deg, 
        R_tm, T_tm, A_tm, 
        R_te, T_te, A_te
    ))
    
    header_text = 'Angle(deg)\tR_TM\tT_TM\tA_TM\tR_TE\tT_TE\tA_TE'
    
    csv_buffer = io.BytesIO()
    np.savetxt(csv_buffer, data_matrix, fmt='%.6f', delimiter='\t', header=header_text, comments='')
    
    st.download_button(
        label="📥 Download File Data_Fresnel_TMM_Complete.txt",
        data=csv_buffer.getvalue(),
        file_name="Data_Fresnel_TMM_Complete.txt",
        mime="text/plain"
    )
    
