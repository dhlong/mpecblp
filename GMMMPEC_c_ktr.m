function [cineq, c, dcineq, dc] = GMMMPEC_c_ktr(x0) 

% GMMMPEC_c
% Constraints for the random coefficients Logit estimated via MPEC.
%
% source: Dube, Fox and Su (2012)
% Code Revised: January 2012


global x IV K prods T v x nn share 
global prodsMarket numProdsTotal marketStarts marketEnds

theta1 = x0(1:K+1, 1);                              % mean tastes
theta2 = x0(K+2:2*K+2, 1);                          % st. deviation of tastes
delta = x0(2*K+3:2*K+2+numProdsTotal, 1);                 % mean utilities
g = x0(2*K+3+numProdsTotal:end, 1);                       % moment condition values

cong = g - IV'*(delta - x*theta1);  % constraints on moment conditions

expmu = exp(x*diag(theta2)*v);      % exponentiated deviations from mean utilities
expmeanval = exp(delta);
[EstShare, simShare] = ind_shnormMPEC(expmeanval,expmu);

cineq = [];
dcineq = [];

c = [EstShare - share ;
     cong ]; 
 
if nargout>2,

    nx0 = size(x0,1);
    ng = size(g,1);
    ooo = ones(1,K+1);
    
    % Evaluate the Gradients
    dSdtheta2 = zeros(numProdsTotal,K+1);
    dSddeltaDIAG = zeros(numProdsTotal,prods);
    dSddelta = zeros(numProdsTotal, numProdsTotal);
    
    for t=1:T,
        index = marketStarts(t):marketEnds(t);
        ooo1 = ones(prodsMarket(t),1);
        for rr = 1:nn,
            dSddeltaDIAG(index,1:prodsMarket(t)) = dSddeltaDIAG(index,1:prodsMarket(t)) + (diag(simShare(index,rr)) - simShare(index,rr)*simShare(index,rr)')/nn;
            dSdtheta2(index,:) = dSdtheta2(index,:) + (simShare(index,rr)*ooo).*(ooo1*v(:,rr)').*( x(index,:) - (ooo1*(simShare(index,rr)'*x(index,:))))/nn;
        end
        dSddelta(index,index) = dSddeltaDIAG(index,1:prodsMarket(t));
    end


    dc1 = [zeros(numProdsTotal,K+1), dSdtheta2, dSddelta, zeros(numProdsTotal,ng)];
    dc2 = [IV'*x, zeros(ng, K+1), -IV', eye(ng)];
    dc = [dc1; dc2]';
    
end