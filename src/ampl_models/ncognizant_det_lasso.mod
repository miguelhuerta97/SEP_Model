set BUSES;
set BUSES_dummy;

param eps_lb := 0.0001;
param R {BUSES, BUSES_dummy};
param B {BUSES, BUSES_dummy};
param Hu {BUSES};
param cost_lasso, default 0.01;
param cost_gamma, default 0.01;

var t >= 0;
var gp {BUSES};
var gq {BUSES};
var v_del_pos {BUSES}, >= 0.;
var v_del_neg {BUSES}, >= 0.;
var eps >= eps_lb, <= 1.;

minimize P1_14a: t - cost_gamma * eps - cost_lasso * sum{i in BUSES} (gp[i] + gq[i]);

s.t. INF_NORM_POS{i in BUSES}: t >= v_del_pos[i] ;
s.t. INF_NORM_NEG{i in BUSES}: t >= v_del_neg[i] ;

s.t. P1_14b{i in BUSES}:
	v_del_pos[i] - v_del_neg[i]
	==
	Hu[i] + sum{k in BUSES} ((R[i,k]*gp[k] + B[i, k]*gq[k]) * Hu[k])
;

s.t. P1_14c:
	sum{i in BUSES } sum{j in BUSES} (gp[i] * R[i,j] + gq[i] * B[i,j])^2
	<=
	1 - eps
;

s.t. P1_14e_p{i in BUSES}:
	gp[i] <= 0
;

s.t. P1_14e_q{i in BUSES}:
	gq[i] <= 0
;
