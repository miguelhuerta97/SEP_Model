/******************************************************************************/
/**                                  SETS                                    **/
/******************************************************************************/
set BUSES0;
set BUSES;
set BUSES_dummy;
set DEMANDS;
set PERIODS;
set DGS;
/******************************************************************************/
/**                               PARAMETERS                                 **/
/******************************************************************************/
## Param defining sets
param n_deg, >=0;
set DEGREES := 0..n_deg;
param fact{n in DEGREES} = if n <= 1 then 1 else fact[n - 1] * n;

## System
param eps_lb := 0.0001;
param slack_bus, integer;

## Objective function coefficients
param cost_nse >= 0;
param cost_putility >= 0;
param cost_vlim >= 0;
param cost_losses >= 0;
param cost_lasso_v >= 0;
param cost_lasso_x >= 0;
param cost_gamma, default 0.01;

## Buses
param busName {BUSES0}, symbolic;
param busVmin {BUSES0};
param busVmax {BUSES0};
param Vnom {BUSES} >= 0.;

## PV generators
param dgName	{DGS}, symbolic;
param dgBus		{DGS}, integer;
param dgSnom	{DGS};

## BUSES
param lFrom		{BUSES}, integer;
param lTo		{BUSES}, integer;
param lR		{BUSES};
param lX		{BUSES};
param lB		{BUSES};
param lImax		{BUSES};

param R {BUSES, BUSES_dummy};
param X {BUSES, BUSES_dummy};

## Demands
param dBus		{DEMANDS};

## Time series
param tsDemandP	{PERIODS, DEMANDS};  # Demand active power
param tsDemandQ {PERIODS, DEMANDS};  # Demand reactive power
param tsPmax 	{PERIODS, DGS}, >= 0;	  # DGs maximum available power

## Nominal setpoints
param dgp_nom {DGS};
param dgq_nom {DGS};

param mean_tsDemandP {DEMANDS};
param mean_tsDemandQ {DEMANDS};
param std_tsDemandP {DEMANDS};
param std_tsDemandQ {DEMANDS};
/******************************************************************************/
/**                                VARIABLES                                 **/
/******************************************************************************/
var ep {BUSES, PERIODS}, >= 0;
var em {BUSES, PERIODS}, >= 0;

var V {BUSES0, PERIODS}, >= 0;
var P {BUSES, PERIODS};
var Q {BUSES, PERIODS};

var nse {BUSES, PERIODS}, >= 0;

var dgp {DGS, PERIODS}, >= 0;
var dgq {DGS, PERIODS};

var dg_Gpv_neg {DGS}, >= 0;
var dg_Gqv_neg {DGS}, >= 0;

var dg_Gpx_pos {DGS, DEGREES}, >= 0;
var dg_Gpx_neg {DGS, DEGREES}, >= 0;
var dg_Gqx_pos {DGS, DEGREES}, >= 0;
var dg_Gqx_neg {DGS, DEGREES}, >= 0;

var eps >= eps_lb, <= 1.;

/******************************************************************************/
/**                                  MODEL                                   **/
/******************************************************************************/
minimize FOBJ:
	+ cost_putility * sum{t in PERIODS} (
	        sum{i in BUSES: lTo[i] == slack_bus} - P[i, t]
	    )
	+ cost_losses * sum{i in BUSES, t in PERIODS} lR[i] * (P[i, t]^2 + Q[i, t]^2)
	+ cost_vlim * sum{i in BUSES, t in PERIODS} (ep[i, t] + em[i,t])
	+ cost_nse * sum{i in BUSES, t in PERIODS} nse[i, t]
	+ cost_lasso_v * sum{i in DGS} (dg_Gpv_neg[i] + dg_Gqv_neg[i])
	+ cost_lasso_x * sum{i in DGS, j in DEGREES} (
	        dg_Gpx_pos[i, j] + dg_Gpx_neg[i, j] + dg_Gqx_pos[i, j] + dg_Gqx_neg[i, j]
	    )
	- cost_gamma * eps
;

# Voltage limits
s.t. V0LIMS{t in PERIODS}: V[slack_bus,t] == 1.0;	# fixed
s.t. VUB{i in BUSES, t in PERIODS}:
	V[i, t] <= busVmax[i] + ep[i, t];
s.t. VLB{i in BUSES, t in PERIODS}:
	busVmin[i] - em[i, t] <= V[i, t];

# Node balance
s.t. PBALANCE{i in BUSES, t in PERIODS}:
	+ sum{j in DGS: dgBus[j] == i} (dgp[j, t])
	- sum{j in DEMANDS: dBus[j] == i} (tsDemandP[t, j])
	+ nse[i, t]
	==
	+ sum{j in BUSES: lFrom[j] == i} P[j, t]
	- sum{j in BUSES: lTo[j] == i} P[j, t]
;

s.t. QBALANCE{i in BUSES, t in PERIODS}:
	+ sum{j in DGS: dgBus[j] == i} (dgq[j, t])
	- sum{j in DEMANDS: dBus[j] == i} tsDemandQ[t, j]
	==
	+ sum{j in BUSES: lFrom[j] == i} (Q[j, t])
	- sum{j in BUSES: lTo[j] == i} (Q[j, t])
;

# Power flow LinDistFlow
s.t. FLUX_LINDISTFLOW {i in BUSES, t in PERIODS}:
    V[lTo[i], t]
	==
	V[lFrom[i],t] - lR[i]*P[i, t] - lX[i]*Q[i, t]
;

## Line flux bounds LinDistFlow
s.t. FLUX_UB_LINDISTFLOW{i in BUSES, t in PERIODS}:
    P[i, t] ^ 2 + Q[i, t] ^ 2 <= lImax[i]^2
;

## PV feasible dispatch
s.t. PV_SNOM {i in DGS, t in PERIODS}:
	dgp[i, t] ^ 2 + dgq[i, t]^2 <= dgSnom[i]^2
;
s.t. PV_PMAX {i in DGS, t in PERIODS}:
	dgp[i, t] <= tsPmax[t, i]
;

## Control dgp
s.t. CTRL_DGP{i in DGS, t in PERIODS}:
    dgp[i, t]
    ==
    # Feedback
    + (- dg_Gpv_neg[i]) * (V[dgBus[i], t] - Vnom[dgBus[i]])
    # Feedforward
    + sum{k in DEGREES} (
            (1 / fact[k]) * (dg_Gpx_pos[i, k] - dg_Gpx_neg[i, k]) *
            ((tsDemandP[t, dgBus[i]] - mean_tsDemandP[dgBus[i]]) / std_tsDemandP[dgBus[i]])^k
        )
    # Nominal value
    + dgp_nom[i]
;

## Control dgq
s.t. CTRL_DGQ{i in DGS, t in PERIODS}:
    dgq[i, t]
    ==
    # Feedback
    + (- dg_Gqv_neg[i]) * (V[dgBus[i], t] - Vnom[dgBus[i]])
    # Feedforward
    + sum{k in DEGREES} (
            (1 / fact[k]) * (dg_Gqx_pos[i, k] - dg_Gqx_neg[i, k]) *
            ((tsDemandQ[t, dgBus[i]] - mean_tsDemandQ[dgBus[i]]) / std_tsDemandQ[dgBus[i]])^k
        )
    # Nominal value
    + dgq_nom[i]
;

## Stability constraints
s.t. STABILITY:
    sum {i in BUSES, j in DGS} (R[i, dgBus[j]] * dg_Gpv_neg[j] + X[i, dgBus[j]] * dg_Gqv_neg[j])^2
	<=
	1 - eps
;
