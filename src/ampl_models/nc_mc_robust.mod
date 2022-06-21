param n_deg >= 0, integer;

set BUSES;
set LINKS within (BUSES cross BUSES);
set PERIODS;
set DEGS := 0..n_deg;

param cost_stability, >= 0;
param cost_lasso_v >= 0;
param cost_lasso_x >= 0;
param cost_putility >= 0;
param cost_losses >= 0;
param cost_vlim >= 0;

param R {LINKS};
param X {LINKS};

param v_lb {BUSES};
param v_ub {BUSES};
param v_set {BUSES};

param dv_lb {BUSES};
param dv_ub {BUSES};

param gp_lb {BUSES};
param gq_lb {BUSES};

param smax {BUSES};
param up_max {BUSES, PERIODS};
param dp_nc {BUSES, PERIODS};
param dq_nc {BUSES, PERIODS};
param up_max_scaled {BUSES, PERIODS}, default 0;
param dp_nc_scaled {BUSES, PERIODS}, default 0;
param dq_nc_scaled {BUSES, PERIODS}, default 0;
param up_mean{BUSES};  # Actuator nominal active power dispatch
param uq_mean{BUSES};  # Actuator nominal reactive power dispatch
param p_mean{BUSES};  # Bus net active power injection
param q_mean{BUSES};  # Bus net reactive power injection

var epi_aux, >= 0;  # Auxiliary variable for epigraph formulation
var eps, >= 0;      # Stability margin

var dv_pos {BUSES, PERIODS}, >=0;
var dv_neg {BUSES, PERIODS}, >=0;

# McCormick auxiliary variables
var k_p_pos {BUSES, PERIODS};
var k_p_neg {BUSES, PERIODS};
var k_q_pos {BUSES, PERIODS};
var k_q_neg {BUSES, PERIODS};

# Ctrl coefficients
var gp {BUSES}, <= 0;
var gq {BUSES}, <= 0;

var DL_pp_pos {BUSES, DEGS}, >= 0;
var DL_pq_pos {BUSES, DEGS}, >= 0;
var DL_qp_pos {BUSES, DEGS}, >= 0;
var DL_qq_pos {BUSES, DEGS}, >= 0;
var DL_pp_neg {BUSES, DEGS}, >= 0;
var DL_pq_neg {BUSES, DEGS}, >= 0;
var DL_qp_neg {BUSES, DEGS}, >= 0;
var DL_qq_neg {BUSES, DEGS}, >= 0;

var DP_p_pos {BUSES, DEGS}, >= 0;
var DP_q_pos {BUSES, DEGS}, >= 0;
var DP_p_neg {BUSES, DEGS}, >= 0;
var DP_q_neg {BUSES, DEGS}, >= 0;

# Auxiliary variables
var dup {BUSES, PERIODS};
var duq {BUSES, PERIODS};

var e_pos {BUSES, PERIODS};
var e_neg {BUSES, PERIODS};


minimize FOBJ:
    + epi_aux
    - cost_stability * eps
    - cost_lasso_v * sum {i in BUSES} (gp[i] + gq[i])
;

minimize PROPOSED_FOBJ:
    + epi_aux
    - cost_stability * eps
    - cost_lasso_v * sum {i in BUSES} (gp[i] + gq[i])
    + cost_lasso_x * sum {i in BUSES, k in DEGS} (
        DL_pp_pos[i,k] + DL_pq_pos[i,k] + DL_qp_pos[i,k] + DL_qq_pos[i,k] + DL_pp_neg[i,k] +
        DL_pq_neg[i,k] + DL_qp_neg[i,k] + DL_qq_neg[i,k] + DP_p_pos[i,k] + DP_q_pos[i,k] +
        DP_p_neg[i,k] + DP_q_neg[i,k]
      )
;

# Epigraph formulation of infinite norm
s.t. PROPOSED_EPIGRAPH{t in PERIODS}:
    epi_aux
    >=
    - cost_putility * sum{i in BUSES} dup[i, t]
    + cost_losses * sum{i in BUSES} (
        (p_mean[i] + dp_nc[i, t] + dup[i, t]) *
        sum{j in BUSES} (
            R[i, j] * (p_mean[j] + dp_nc[j, t] + dup[j, t])
        )
        +
        (q_mean[i] + dq_nc[i, t] + duq[i, t]) *
        sum{j in BUSES} (
            R[i, j] * (q_mean[j] + dq_nc[j, t] + duq[j, t])
        )
    )
    + cost_vlim * sum{i in BUSES} (e_pos[i, t] + e_neg[i, t])
;

s.t. INFINITE_NORM_UB {i in BUSES, t in PERIODS}: epi_aux >= dv_pos[i, t];
s.t. INFINITE_NORM_LB {i in BUSES, t in PERIODS}: epi_aux >= dv_neg[i, t];

# Voltage bounds
s.t. VOLTAGE_LB {i in BUSES, t in PERIODS}: dv_lb[i] <= - dv_neg[i, t];
s.t. VOLTAGE_UB {i in BUSES, t in PERIODS}: dv_ub[i] >= dv_pos[i, t];

s.t. VOLTAGE_LB2 {i in BUSES, t in PERIODS}:
    v_lb[i] - e_neg[i, t] <= v_set[i] + dv_pos[i, t] - dv_neg[i, t]
;
s.t. VOLTAGE_UB2 {i in BUSES, t in PERIODS}:
    v_ub[i] + e_pos[i, t] >= v_set[i] + dv_pos[i, t] - dv_neg[i, t]
;

# Ctrl coefficients bounds
s.t. GP_LB {i in BUSES}: gp[i] >= gp_lb[i];
s.t. GQ_LB {i in BUSES}: gq[i] >= gq_lb[i];

# Stability
s.t. STABILITY:
    sum {(i, j) in LINKS} ((R[i, j] * gp[j] + X[i, j] * gq[j])^2) <= 1 - eps
;

# Linear power flow model (Bilinear formulation)
s.t. BILINEAR_PFLOW {i in BUSES, t in PERIODS}:
    dv_pos[i, t] - dv_neg[i, t]
    ==
    sum {j in BUSES} (
        # Controlled
        + R[i, j] * gp[j] * (dv_pos[j, t] - dv_neg[j, t])
        + X[i, j] * gq[j] * (dv_pos[j, t] - dv_neg[j, t])
        # Uncontrolled
        + R[i, j] * dp_nc[j, t] + X[i, j] * dq_nc[j, t]
    )
;

# Linear power flow model (Neuman formulation)
s.t. NEUMAN_PFLOW {i in BUSES, t in PERIODS}:
    dv_pos[i, t] - dv_neg[i, t]
    ==
    sum {j in BUSES} (
        + (R[i, j] * gp[j] + X[i, j] * gp[j])
        * (R[i, j] * dp_nc[j, t] + X[i, j] * dq_nc[j, t])
        + (R[i, j] * dp_nc[j, t] + X[i, j] * dq_nc[j, t])
    )
;

# Linear power flow model (McCormick formulation)
s.t. MCCORMICK_PFLOW {i in BUSES, t in PERIODS}:
    dv_pos[i, t] - dv_neg[i, t]
    ==
    sum{j in BUSES} (
        # Controlled part
        + R[i, j] * dup[j, t]
        + X[i, j] * duq[j, t]
        # Uncontrolled part
        + R[i, j] * dp_nc[j, t] + X[i, j] * dq_nc[j, t]
    )
;

s.t. PROPOSED_CONTROL_P{j in BUSES, t in PERIODS}:
    dup[j, t]
    ==
    + k_p_pos[j, t] - k_p_neg[j, t]
    + sum{k in DEGS} (
        # \xi_L laws
        + (DL_pp_pos[j, k] - DL_pp_neg[j, k]) * (dp_nc_scaled[j, t])^k
        + (DL_pq_pos[j, k] - DL_pq_neg[j, k]) * (dq_nc_scaled[j, t])^k
        # \xi_P laws
        + (DP_p_pos[j, k] - DP_p_neg[j, k]) * (up_max_scaled[j, t])^k
    )
;

s.t. PROPOSED_CONTROL_Q{j in BUSES, t in PERIODS}:
    duq[j, t]
    ==
    + k_q_pos[j, t] - k_q_neg[j, t]
    + sum{k in DEGS} (
        # \xi_L laws
        + (DL_qp_pos[j, k] - DL_qp_neg[j, k]) * (dp_nc_scaled[j, t])^k
        + (DL_qq_pos[j, k] - DL_qq_neg[j, k]) * (dq_nc_scaled[j, t])^k
        # \xi_D laws
        + (DP_q_pos[j, k] - DP_q_neg[j, k]) * (up_max_scaled[j, t])^k
    )
;

# OID chart
s.t. PROPOSED_SMAX{i in BUSES, t in PERIODS}:
    (up_mean[i] + dup[i, t])^2 + (uq_mean[i] + duq[i, t])^2 <= smax[i]^2
;

s.t. PROPOSED_PMAX{i in BUSES, t in PERIODS}:
    up_mean[i] + dup[i, t] <= up_max[i, t]
;

# McCormick envelopes autogenerated
s.t. MCCORMICK_P_POS0{i in BUSES, t in PERIODS}:
-k_p_pos[i, t] + gp_lb[i]*dv_pos[i, t] + 0*gp[i] <= gp_lb[i]*0
;
s.t. MCCORMICK_P_POS1{i in BUSES, t in PERIODS}:
-k_p_pos[i, t] + 0*dv_pos[i, t] + dv_ub[i]*gp[i] <= 0*dv_ub[i]
;
s.t. MCCORMICK_P_POS2{i in BUSES, t in PERIODS}:
-k_p_pos[i, t] + 0*dv_pos[i, t] + 0*gp[i] >= 0
;
s.t. MCCORMICK_P_POS3{i in BUSES, t in PERIODS}:
-k_p_pos[i, t] + gp_lb[i]*dv_pos[i, t] + dv_ub[i]*gp[i] >= gp_lb[i]*dv_ub[i]
;

s.t. MCCORMICK_P_NEG0{i in BUSES, t in PERIODS}:
-k_p_neg[i, t] + gp_lb[i]*dv_neg[i, t] + 0*gp[i] <= gp_lb[i]*0
;
s.t. MCCORMICK_P_NEG1{i in BUSES, t in PERIODS}:
-k_p_neg[i, t] + 0*dv_neg[i, t] + -dv_lb[i]*gp[i] <= 0*-dv_lb[i]
;
s.t. MCCORMICK_P_NEG2{i in BUSES, t in PERIODS}:
-k_p_neg[i, t] + 0*dv_neg[i, t] + 0*gp[i] >= 0
;
s.t. MCCORMICK_P_NEG3{i in BUSES, t in PERIODS}:
-k_p_neg[i, t] + gp_lb[i]*dv_neg[i, t] + -dv_lb[i]*gp[i] >= gp_lb[i]*-dv_lb[i]
;

s.t. MCCORMICK_Q_POS0{i in BUSES, t in PERIODS}:
-k_q_pos[i, t] + gq_lb[i]*dv_pos[i, t] + 0*gq[i] <= gq_lb[i]*0
;
s.t. MCCORMICK_Q_POS1{i in BUSES, t in PERIODS}:
-k_q_pos[i, t] + 0*dv_pos[i, t] + dv_ub[i]*gq[i] <= 0*dv_ub[i]
;
s.t. MCCORMICK_Q_POS2{i in BUSES, t in PERIODS}:
-k_q_pos[i, t] + 0*dv_pos[i, t] + 0*gq[i] >= 0
;
s.t. MCCORMICK_Q_POS3{i in BUSES, t in PERIODS}:
-k_q_pos[i, t] + gq_lb[i]*dv_pos[i, t] + dv_ub[i]*gq[i] >= gq_lb[i]*dv_ub[i]
;

s.t. MCCORMICK_Q_NEG0{i in BUSES, t in PERIODS}:
-k_q_neg[i, t] + gq_lb[i]*dv_neg[i, t] + 0*gq[i] <= gq_lb[i]*0
;
s.t. MCCORMICK_Q_NEG1{i in BUSES, t in PERIODS}:
-k_q_neg[i, t] + 0*dv_neg[i, t] + -dv_lb[i]*gq[i] <= 0*-dv_lb[i]
;
s.t. MCCORMICK_Q_NEG2{i in BUSES, t in PERIODS}:
-k_q_neg[i, t] + 0*dv_neg[i, t] + 0*gq[i] >= 0
;
s.t. MCCORMICK_Q_NEG3{i in BUSES, t in PERIODS}:
-k_q_neg[i, t] + gq_lb[i]*dv_neg[i, t] + -dv_lb[i]*gq[i] >= gq_lb[i]*-dv_lb[i]
;

