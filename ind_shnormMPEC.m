function [EstShare, simShare] = ind_shnormMPEC(expmeanval,expmu)

% IND_SHNORM
% This function computes the distribution of "individual" probabilities of choosing each brand.
% The probabilities are those associated with the normally-distributed r.c. logit.
%
% ARGUMENTS:
% expmeanval = vector of exponentiated mean utilities, sorted by market then product
% expmu = matrix of exponentiated draws of deviations from mean utilities, sorted by market then product
%
% OUTPUT:
% PROB = vector of expected market shares, sorted by market then product
%
% source: Dube, Fox and Su (2012)
% Code Revised: January 2012

global oo sharesum marketForProducts

numer = (expmeanval*oo ).*expmu;        % this is the numerator (oo speeds-up expanding mean utility by number of draws)
sum1 = sharesum*numer;
sum11 = 1./(1+sum1);                    % this is the denominator of the shares
denom1 = sum11(marketForProducts,:);          % this expands the denominator
simShare = numer.*denom1;               % simulated shares for each draw
EstShare = mean(simShare,2);            % expected share (i.e. mean across draws)
