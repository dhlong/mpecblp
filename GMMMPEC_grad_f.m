function grad_f = GMMMPEC_grad_f(x0)

% GMMMPEC_f
% GMM Objective function for the random coefficients Logit esitmated via
% MPEC.
%
% source: Dube, Fox and Su (2012)
% Code Revised: January 2012


global W K numProdsTotal

g = x0(2*K+3+numProdsTotal:end, 1);               % moment condition values
nx0 = size(x0,1);
grad_f = zeros(nx0,1);
grad_f(2*K+3+numProdsTotal:end,1) = 2*W*g;
