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
param cost_putility >= 0;
param cost_vlim >= 0;
param sto_weights {PERIODS}, default 1;

## Buses
param busVmin {BUSES0};
param busVmax {BUSES0};

## PV generators
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

var dgp {DGS}, >= 0;
var dgq {DGS};

var b2btp {B2BS};
var b2btq {B2BS};
var b2bfp {B2BS};
var b2bfq {B2BS};

/******************************************************************************/
/**                                  MODEL                                   **/
/******************************************************************************/
minimize FOBJ:
    sum{t in PERIODS} sto_weights[t] * (
        cost_putility * (
            - sum{j in LINES: lTo[j] == slack_bus} (P[j, t] - lR[j] * l[j, t])
        )
        + cost_vlim * sum{i in BUSES} (ep[i, t] + em[i, t])
	)
;

# Voltage limits
s.t. V0LIMS{t in PERIODS}: v[slack_bus,t] == 1.0;	# fixed
s.t. VUB{i in BUSES, t in PERIODS}:
	v[i, t] <= busVmax[i] ** 2 + ep[i, t];
s.t. VLB{i in BUSES, t in PERIODS}:
	busVmin[i] ** 2 - em[i, t] <= v[i, t];

# Node balance
s.t. PBALANCE{i in BUSES, t in PERIODS}:
	+ sum{j in DGS: dgBus[j] == i} (dgp[j])
	+ sum{j in B2BS: b2bFrom[j] == i} (b2bfp[j])
	+ sum{j in B2BS: b2bTo[j] == i} (b2btp[j])
	- sum{j in DEMANDS: dBus[j] == i} (tsDemandP[t, j])
	==
	+ sum{j in LINES: lFrom[j] == i} P[j, t]
	- sum{j in LINES: lTo[j] == i} (P[j, t] - lR[j] * l[j, t])
;

s.t. QBALANCE{i in BUSES, t in PERIODS}:
	+ sum{j in DGS: dgBus[j] == i} (dgq[j])
	+ sum{j in B2BS: b2bFrom[j] == i} (b2bfq[j])
	+ sum{j in B2BS: b2bTo[j] == i} (b2btq[j])
	- sum{j in DEMANDS: dBus[j] == i} tsDemandQ[t, j]
	==
	+ sum{j in LINES: lFrom[j] == i} (Q[j, t] - lB[j] * v[i, t])
	- sum{j in LINES: lTo[j] == i} (Q[j, t] - lX[j] * l[j, t] + lB[j] * v[i, t])
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
s.t. PV_SNOM {i in DGS}:
	dgp[i] ^ 2 + dgq[i]^2 <= dgSnom[i]^2
;
s.t. PV_PMAX {i in DGS, t in PERIODS}:
	dgp[i] <= tsPmax[t, i]
;
