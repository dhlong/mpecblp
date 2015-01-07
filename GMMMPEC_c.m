function c = GMMMPEC_c(x0) 

% GMMMPEC_c
% Constraints for the random coefficients Logit estimated via MPEC.
%
% source: Dube, Fox and Su (2012)
% Code Revised: January 2012


global x IV K v  share 
global numProdsTotal

theta1 = x0(1:K+1, 1);                              % mean tastes
theta2 = x0(K+2:2*K+2, 1);                          % st. deviation of tastes
delta = x0(2*K+3:2*K+2+numProdsTotal, 1);                 % mean utilities
g = x0(2*K+3+numProdsTotal:end, 1);                       % moment condition values

cong = g - IV'*(delta - x*theta1);  % constraints on moment conditions

expmu = exp(x*diag(theta2)*v);      % exponentiated deviations from mean utilities
expmeanval = exp(delta);
EstShare = ind_shnormMPEC(expmeanval,expmu);

c = [EstShare - share ;
     cong ]; 

end