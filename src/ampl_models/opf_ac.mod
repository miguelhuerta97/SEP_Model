/******************************************************************************/
/**                                  SETS                                    **/
/******************************************************************************/
set BUSES0 ;
set BUSES ;
set DEMANDS ;
set DGS ;

/******************************************************************************/
/**                               PARAMETERS                                 **/
/******************************************************************************/
## System
param SLACK_BUS;
param SBase_MVA ;
param cost_putility >= 0;
param cost_vlim >= 0;

## Buses
param busVmin {BUSES};
param busVmax {BUSES};

## PV generators
param dgBus		{DGS}, integer ;
param dgSnom	{DGS} ;

## Lines
param lTo       {BUSES} ;
param ly		{BUSES} ;
param ltheta	{BUSES} ;

## Demands
param dBus		{DEMANDS};

## Time series
param loadp {DEMANDS};
param loadq {DEMANDS};
param dgpmax {DGS};

param pi := 3.141592653589793;
/******************************************************************************/
/**                                VARIABLES                                 **/
/******************************************************************************/
var V {BUSES0}, >= 0.8, <= 1.2;
var theta {BUSES0}, >=-pi, <= pi;

var ep {BUSES}, >= 0;
var em {BUSES}, >= 0;

var dgp {DGS}, >= 0;
var dgq {DGS};

var P_up {BUSES};
var Q_up {BUSES};

var P_down {BUSES};
var Q_down {BUSES};

/******************************************************************************/
/**                                  MODEL                                   **/
/******************************************************************************/
minimize FOBJ:
    # Energy power injections
    cost_putility * sum {k in BUSES: lTo[k] == SLACK_BUS} P_up[k]
    # Voltage limits
    + cost_vlim * sum {i in BUSES} (ep[i] + em[i])
;

s.t. SLACK: V[SLACK_BUS] == 1;
s.t. SLACK2: theta[SLACK_BUS] == 0;

s.t. VUB{i in BUSES}:
	V[i] <= busVmax[i] + ep[i];

s.t. VLB{i in BUSES}:
	busVmin[i] - em[i] <= V[i];

s.t. DEF_P_up {j in BUSES}:

    P_up[j]
    ==
    V[lTo[j]] * V[j] * ly[j] * cos(theta[lTo[j]] - theta[j] - ltheta[j])
    - (V[lTo[j]]^2) * ly[j] * cos(ltheta[j])
;

s.t. DEP_Q_up {j in BUSES}:
    Q_up[j]
    ==
    V[lTo[j]] * V[j] * ly[j] * sin(theta[lTo[j]] - theta[j] - ltheta[j])
    - (V[lTo[j]]^2) * ly[j] * sin(-ltheta[j])
;

s.t. DEP_P_down {j in BUSES}:
    P_down[j]
    ==
    (V[j]^2) * ly[j] * cos(ltheta[j])
    - V[j] * V[lTo[j]] * ly[j] * cos(theta[j] - theta[lTo[j]] - ltheta[j])
;

s.t. DEP_Q_down {j in BUSES}:
    Q_down[j]
    ==
    (V[j]^2) * ly[j] * sin(-ltheta[j])
    - V[j] * V[lTo[j]] * ly[j] * sin(theta[j] - theta[lTo[j]] - ltheta[j])
;

s.t. P_BALANCE{j in BUSES}:
	+ sum{k in DGS: dgBus[k] == j} dgp[k]
	- sum{k in DEMANDS: dBus[k] == j} loadp[k]
	==
    P_down[j] - sum{k in BUSES: lTo[k] == j} P_up[k]
;

s.t. Q_BALANCE{j in BUSES}:
	+ sum{k in DGS: dgBus[k] == j} dgq[k]
	- sum{k in DEMANDS: dBus[k] == j} loadq[k]
	==
	Q_down[j] - sum {k in BUSES: lTo[k] == j} Q_up[k]
;

# PV feasible dispatch
s.t. PV_SNOM {i in DGS}:
	dgp[i] ^ 2 + dgq[i]^2 <= dgSnom[i]^2
;
s.t. PV_PMAX {i in DGS}:
	dgp[i] <= dgpmax[i]
;