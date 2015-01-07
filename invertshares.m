function delta = invertshares(theta2)

% INVERTSHARES
% GMM objective function for the random coefficients Logit estimated via
% the NFP approach proposed by BLP.
%
% source: Dube, Fox and Su (2010)
% Code Revised: March 2009

global x expmeanval tol_inner share v
global delta

ii = 0;
norm_max = 1;
expmeanval0 = expmeanval;

expmu = exp(x*diag(theta2)*v);      % exponentiated deviations from mean utilities

while norm_max > tol_inner && ii < 2500,
    expmeanval1 = expmeanval0.*share./ind_shnorm(expmeanval0,expmu); 
    t = abs(expmeanval1-expmeanval0);
    norm_max = max(t);
    expmeanval0 = expmeanval1;
    ii = ii + 1;
end;

if max(isnan(expmeanval0))<1, expmeanval = expmeanval0; end
delta = log(expmeanval);               % the mean utilities
