function [hess] = GMMMPEC_hess_ktr(x0,lambda) 

% GMMMPEC_c
% Constraints for the random coefficients Logit estimated via MPEC.
%
% source: Dube, Fox and Su (2012)
% Code Revised: January 2012


global x IV K prods T v x nn logshare share W 
global prodsMarket numProdsTotal marketStarts marketEnds

theta1 = x0(1:K+1, 1);                              % mean tastes
theta2 = x0(K+2:2*K+2, 1);                          % st. deviation of tastes
delta = x0(2*K+3:2*K+2+numProdsTotal, 1);                 % mean utilities
g = x0(2*K+3+numProdsTotal:end, 1);                       % moment condition values

nx0 = size(x0,1);
ng = size(g,1);
ooo = ones(1,K+1);


hess = zeros(nx0, nx0);
hess(2*K+3+numProdsTotal:end, 2*K+3+numProdsTotal:end) = 2*W;

expmu = exp(x*diag(theta2)*v);      % exponentiated deviations from mean utilities
expmeanval = exp(delta);
[EstShare, simShare] = ind_shnormMPEC(expmeanval,expmu);

dL2dtheta22 = zeros(K+1,K+1);
dL2dtheta22rr = zeros(K+1,K+1);
dL2ddeltadtheta = zeros(numProdsTotal, K+1);
dL2ddelta2DIAG = zeros(numProdsTotal,prods);
dL2ddelta2 = zeros(numProdsTotal, numProdsTotal);

% Evaluate the hessian    
for t=1:T,
    index = marketStarts(t):marketEnds(t);
    multip = lambda.eqnonlin(index);
    ooo1 = ones(prodsMarket(t),1);
    for rr = 1:nn,
            
        simS = simShare(index,rr);
        sumprod_mpsimS = multip'*simS;
            
        blk1 = sumprod_mpsimS*(-diag(simS) + 2*simS*simS');          
        blk2 = -(multip.*simS)*simS';
        blk3 = diag(multip.*simS);
        blk = blk1 + blk2 + blk2'+blk3;        
        dL2ddelta2DIAG(index,1:prodsMarket(t)) = dL2ddelta2DIAG(index,1:prodsMarket(t)) + blk/nn;
            
       
        
        xsimSx = x(index,:) - (ooo1*(simS'*x(index,:)));
        xsimSxv = xsimSx.*(ooo1*v(:,rr)');
        dSdtheta2rr = (simS*ooo).*xsimSxv;
     
        dL2ddeltadthetarr = -simS*multip'*dSdtheta2rr - sumprod_mpsimS*dSdtheta2rr + (multip*ooo).*dSdtheta2rr;       
        dL2ddeltadtheta(index,:) = dL2ddeltadtheta(index,:) + dL2ddeltadthetarr/nn;
        dL2dtheta22rr = ((multip*ooo).*dSdtheta2rr)'*xsimSxv + sumprod_mpsimS*(-dSdtheta2rr'*x(index,:).*(ones(K+1,1)*v(:,rr)')) ;
        dL2dtheta22 = dL2dtheta22 + dL2dtheta22rr/nn;
        
    end
    dL2ddelta2(index,index) = dL2ddelta2DIAG(index,1:prodsMarket(t));
end

hess(K+2:2*K+2,K+2:2*K+2) = dL2dtheta22;        
hess(2*K+3:2*K+2+numProdsTotal,K+2:2*K+2) = dL2ddeltadtheta;    
hess(K+2:2*K+2,2*K+3:2*K+2+numProdsTotal) = dL2ddeltadtheta';        
hess(2*K+3:2*K+2+numProdsTotal, 2*K+3:2*K+2+numProdsTotal) = dL2ddelta2; 
    