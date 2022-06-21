/******************************************************************************/
/**                                  SETS                                    **/
/******************************************************************************/
set BUSES0;
set BUSES;
set DEMANDS;
set PERIODS;
set LINES;
set DGS;
set B2BS;

/******************************************************************************/
/**                               PARAMETERS                                 **/
/******************************************************************************/
## System
param slack_bus, integer;
param cost_nse >= 0;
param cost_putility >= 0;
param cost_vlim >= 0;
param cost_losses >= 0;

## Buses
param busName {BUSES0}, symbolic;
param busVmin {BUSES0};
param busVmax {BUSES0};

## PV generators
param dgName	{DGS}, symbolic;
param dgBus		{DGS}, integer;
param dgSnom	{DGS};

## B2Bs
param b2bFrom	{B2BS}, integer;
param b2bTo		{B2BS}, integer;
param b2bSnom	{B2BS};

## Lines
param lFrom		{LINES}, integer;
param lTo		{LINES}, integer;
param lR		{LINES};
param lX		{LINES};
param lB		{LINES};
param lImax		{LINES};

## Demands
param dBus		{DEMANDS};

## Time series
param tsDemandP	{PERIODS, DEMANDS};  # Demand active power
param tsDemandQ {PERIODS, DEMANDS};  # Demand reactive power
param tsPmax 	{PERIODS, DGS}, >= 0;	  # DGs maximum available power

/******************************************************************************/
/**                                VARIABLES                                 **/
/******************************************************************************/
var ep {BUSES, PERIODS}, >= 0;
var em {BUSES, PERIODS}, >= 0;

var v {BUSES0, PERIODS}, >= 0;
var l {LINES, PERIODS}, >= 0;
var P {LINES, PERIODS};
var Q {LINES, PERIODS};

var nse {BUSES, PERIODS}, >= 0;

var dgp {DGS, PERIODS}, >= 0;
var dgq {DGS, PERIODS};

var b2btp {B2BS, PERIODS};
var b2btq {B2BS, PERIODS};
var b2bfp {B2BS, PERIODS};
var b2bfq {B2BS, PERIODS};

/******************************************************************************/
/**                                  MODEL                                   **/
/******************************************************************************/
minimize FOBJ:
	+ cost_putility * sum{t in PERIODS} (
		sum{i in LINES: lTo[i] == slack_bus} - P[i, t]
	)
	+ cost_losses * sum{i in LINES, t in PERIODS} l[i, t]
	+ cost_vlim * sum{i in BUSES, t in PERIODS} (ep[i, t] + em[i,t])
	+ cost_nse * sum{i in BUSES, t in PERIODS} nse[i, t]
;

minimize FOBJ_LINDISTFLOW:
	+ cost_putility * sum{t in PERIODS} (
		sum{i in LINES: lTo[i] == slack_bus} - P[i, t]
	)
	+ cost_losses * sum{i in LINES, t in PERIODS} (P[i, t]^2 + Q[i, t]^2)
	+ cost_vlim * sum{i in BUSES, t in PERIODS} (ep[i, t] + em[i,t])
	+ cost_nse * sum{i in BUSES, t in PERIODS} nse[i, t]
;


# Voltage limits
s.t. V0LIMS{t in PERIODS}: v[slack_bus,t] == 1.0;	# fixed
s.t. VUB{i in BUSES, t in PERIODS}:
	v[i, t] <= busVmax[i] ** 2 + ep[i, t];
s.t. VLB{i in BUSES, t in PERIODS}:
	busVmin[i] ** 2 - em[i, t] <= v[i, t];

# Node balance
s.t. PBALANCE{i in BUSES, t in PERIODS}:
	+ sum{j in DGS: dgBus[j] == i} (dgp[j, t])
	+ sum{j in B2BS: b2bFrom[j] == i} (b2bfp[j, t])
	+ sum{j in B2BS: b2bTo[j] == i} (b2btp[j, t])
	- sum{j in DEMANDS: dBus[j] == i} (tsDemandP[t, j])
	+ nse[i, t]
	==
	+ sum{j in LINES: lFrom[j] == i} P[j, t]
	- sum{j in LINES: lTo[j] == i} (P[j, t] - lR[j] * l[j, t])
;

s.t. QBALANCE{i in BUSES, t in PERIODS}:
	+ sum{j in DGS: dgBus[j] == i} (dgq[j, t])
	+ sum{j in B2BS: b2bFrom[j] == i} (b2bfq[j, t])
	+ sum{j in B2BS: b2bTo[j] == i} (b2btq[j, t])
	- sum{j in DEMANDS: dBus[j] == i} tsDemandQ[t, j]
	==
	+ sum{j in LINES: lFrom[j] == i} (Q[j, t] - lB[j] * v[i, t])
	- sum{j in LINES: lTo[j] == i} (Q[j, t] - lX[j] * l[j, t] + lB[j] * v[i, t])
;

# Power flow DistFlow
s.t. FLUX_NON_RELAXED {i in LINES, t in PERIODS}:
	P[i, t] ^ 2 + Q[i, t] ^ 2 == l[i, t] * v[lTo[i], t]
;

# Power flow SOCP
s.t. FLUX_RELAXED {i in LINES, t in PERIODS}:
	P[i, t] ^ 2 + Q[i, t] ^ 2 <= l[i, t] * v[lFrom[i], t]
;

s.t. FLUX_EQUALITY {i in LINES, t in PERIODS}:
	v[lTo[i], t]
	==
	v[lFrom[i],t] - 2*lR[i]*P[i, t] - 2*lX[i]*Q[i, t] + (lR[i]^2 + lX[i]^2) * l[i, t]
;

s.t. FLUX_UB1 {i in LINES, t in PERIODS}:
	l[i, t] - 2 * lB[i] * Q[i, t] + (lB[i] ^ 2) * v[lFrom[i], t] <= lImax[i] ^2
;

s.t. FLUX_UB2 {i in LINES, t in PERIODS}:
	l[i, t] + 2 * lB[i] * (Q[i, t] - lX[i] * l[i, t]) + (lB[i]^2) * v[lTo[i], t]
	<=
	lImax[i] ^ 2
;
# Power flow LinDistFlow
s.t. FLUX_LINDISTFLOW {i in LINES, t in PERIODS}:
    v[lTo[i], t]
	==
	v[lFrom[i],t] - 2*lR[i]*P[i, t] - 2*lX[i]*Q[i, t]
;

## Line flux bounds LinDistFlow
s.t. FLUX_UB_LINDISTFLOW{i in LINES, t in PERIODS}:
    P[i, t] ^ 2 + Q[i, t] ^ 2 <= lImax[i]^2
;

# PV feasible dispatch
s.t. PV_SNOM {i in DGS, t in PERIODS}:
	dgp[i, t] ^ 2 + dgq[i, t]^2 <= dgSnom[i]^2
;
s.t. PV_PMAX {i in DGS, t in PERIODS}:
	dgp[i, t] <= tsPmax[t, i]
;

# B2Bs
s.t. B2B_SNOM_F {i in B2BS, t in PERIODS}:
	b2bfp[i, t]^2 + b2bfq[i, t]^2 <= b2bSnom[i]^2
;

s.t. B2B_SNOM_T {i in B2BS, t in PERIODS}:
	b2btp[i, t]^2 + b2btq[i, t]^2 <= b2bSnom[i]^2
;

s.t. B2B_PLINK {i in B2BS, t in PERIODS}:
	b2bfp[i, t] + b2btp[i, t] == 0
;
