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

param dv_ub {BUSES};
param dv_lb {BUSES};

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

param sto_weights {PERIODS}, default 1;

param w_DL_pp {BUSES, DEGS}, default 1;
param w_DL_pq {BUSES, DEGS}, default 1;
param w_DL_qp {BUSES, DEGS}, default 1;
param w_DL_qq {BUSES, DEGS}, default 1;
param w_DP_p {BUSES, DEGS}, default 1;
param w_DP_q {BUSES, DEGS}, default 1;

# Auxiliary epigraph
var epi_ub;

# Voltage limits auxiliary variables
var e_pos {BUSES, PERIODS}, >= 0, <= 0.15;
var e_neg {BUSES, PERIODS}, >= 0, <= 0.15;

# Stability margin
var eps, >= 0;

# State
var dv {BUSES, PERIODS};

# Actuation
var dup {BUSES, PERIODS};
var duq {BUSES, PERIODS};

# Ctrl coefficients
var gp {BUSES}, <= 0;
var gq {BUSES}, <= 0;

var DL_pp_pos {BUSES, DEGS}, >= 0, <= 10;
var DL_pq_pos {BUSES, DEGS}, >= 0, <= 10;
var DL_qp_pos {BUSES, DEGS}, >= 0, <= 10;
var DL_qq_pos {BUSES, DEGS}, >= 0, <= 10;
var DL_pp_neg {BUSES, DEGS}, >= 0, <= 10;
var DL_pq_neg {BUSES, DEGS}, >= 0, <= 10;
var DL_qp_neg {BUSES, DEGS}, >= 0, <= 10;
var DL_qq_neg {BUSES, DEGS}, >= 0, <= 10;

var DP_p_pos {BUSES, DEGS}, >= 0, <= 10;
var DP_q_pos {BUSES, DEGS}, >= 0, <= 10;
var DP_p_neg {BUSES, DEGS}, >= 0, <= 10;
var DP_q_neg {BUSES, DEGS}, >= 0, <= 10;


# Objective function stochastic
minimize FOBJ_STOCHASTIC:
    # Second stage cost
    sum {t in PERIODS} (
        sto_weights[t] * (
            # Losses
            cost_putility * sum{i in BUSES} (
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
            # Negative total variable DG injection
            - cost_putility * sum{i in BUSES} dup[i, t]
            # Cost voltage limits
            + cost_vlim * sum{i in BUSES} (e_neg[i, t] + e_pos[i, t])
        )
    )
    # Stability cost
    - cost_stability * eps
    # Regularization terms
    - cost_lasso_v * sum {i in BUSES} (gp[i] + gq[i])
    + cost_lasso_x * sum {i in BUSES, k in DEGS} (
        + w_DL_pp[i, k] * (DL_pp_pos[i,k] + DL_pp_neg[i,k])
        + w_DL_pq[i, k] * (DL_pq_pos[i,k] + DL_pq_neg[i,k])
        + w_DL_qp[i, k] * (DL_qp_pos[i,k] + DL_qp_neg[i,k])
        + w_DL_qq[i, k] * (DL_qq_pos[i,k] + DL_qq_neg[i,k])
        + w_DP_p[i, k]  * (DP_p_pos[i,k]  + DP_p_neg[i,k])
        + w_DP_q[i, k]  * (DP_q_pos[i,k]  + DP_q_neg[i,k])
      )
;

minimize FOBJ_ROBUST:
    + epi_ub
    # Stability cost
    - cost_stability * eps
    # Regularization terms
    - cost_lasso_v * sum {i in BUSES} (gp[i] + gq[i])
    + cost_lasso_x * sum {i in BUSES, k in DEGS} (
        + w_DL_pp[i, k] * (DL_pp_pos[i,k] + DL_pp_neg[i,k])
        + w_DL_pq[i, k] * (DL_pq_pos[i,k] + DL_pq_neg[i,k])
        + w_DL_qp[i, k] * (DL_qp_pos[i,k] + DL_qp_neg[i,k])
        + w_DL_qq[i, k] * (DL_qq_pos[i,k] + DL_qq_neg[i,k])
        + w_DP_p[i, k]  * (DP_p_pos[i,k]  + DP_p_neg[i,k])
        + w_DP_q[i, k]  * (DP_q_pos[i,k]  + DP_q_neg[i,k])
      )
;

s.t. EPIGRAPH_UB_NEUMAN_1{t in PERIODS, i in BUSES}:
    epi_ub >= cost_vlim * dv[i, t]
;

s.t. EPIGRAPH_UB_NEUMAN_2{t in PERIODS, i in BUSES}:
    - epi_ub <= cost_vlim * dv[i, t]
;

# Epigraph upper bound
s.t. EPIGRAPH_UB{t in PERIODS}:
    epi_ub
    >=
    # Losses
    cost_putility * (
        sum{i in BUSES} (
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
        # Negative total pinjection
        - sum{i in BUSES} (p_mean[i] + dp_nc[i, t] + dup[i, t])
    )
    # Cost voltage limits
    + cost_vlim * sum{i in BUSES} (e_neg[i, t] + e_pos[i, t])
;


# Voltage soft bounds
s.t. VOLTAGE_LB2 {i in BUSES, t in PERIODS}:
    v_lb[i] - e_neg[i, t] <= v_set[i] + dv[i, t]
;
s.t. VOLTAGE_UB2 {i in BUSES, t in PERIODS}:
    v_ub[i] + e_pos[i, t] >= v_set[i] + dv[i, t]
;

# Linear power flow model
s.t. LINEAR_PFLOW {i in BUSES, t in PERIODS}:
    dv[i, t]
    ==
    sum{j in BUSES} (
        # Controlled part
        + R[i, j] * dup[j, t]
        + X[i, j] * duq[j, t]
        # Uncontrolled part
        + R[i, j] * dp_nc[j, t] + X[i, j] * dq_nc[j, t]
    )
;

# Stability
s.t. STABILITY:
    sum {(i, j) in LINKS} ((R[i, j] * gp[j] + X[i, j] * gq[j])^2) <= 1 - eps
;

# Control policy dup
s.t. PROPOSED_CONTROL_P_BILINEAR{j in BUSES, t in PERIODS}:
    dup[j, t]
    ==
    + gp[j] * dv[j, t]
    + sum{k in DEGS} (
        # \xi_L laws
        + (DL_pp_pos[j, k] - DL_pp_neg[j, k]) * (dp_nc_scaled[j, t])^k
        + (DL_pq_pos[j, k] - DL_pq_neg[j, k]) * (dq_nc_scaled[j, t])^k
        # \xi_P laws
        + (DP_p_pos[j, k] - DP_p_neg[j, k]) * (up_max_scaled[j, t])^k
    )
;

# Control policy duq
s.t. PROPOSED_CONTROL_Q_BILINEAR{j in BUSES, t in PERIODS}:
    duq[j, t]
    ==
    + gq[j] * dv[j, t]
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

s.t. PROPOSED_PMIN{i in BUSES, t in PERIODS}:
    up_mean[i] + dup[i, t] >= 0
;