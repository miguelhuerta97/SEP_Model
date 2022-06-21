param pmax, >=0;
param snom, >=0;

param dgp_setpoint, >=0;
param dgq_setpoint;

var dgp, >= 0;
var dgq;

minimize FOBJ: (dgp - dgp_setpoint)^2 + (dgq - dgq_setpoint)^2 ;

s.t. PMAX: dgp <= pmax;
s.t. SNOM: dgp^2 + dgq^2 <= snom^2;
