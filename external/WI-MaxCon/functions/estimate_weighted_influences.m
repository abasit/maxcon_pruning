
function fc = estimate_weighted_influences(fittingfn, nbits, restricted_bits, nsamples, cprob, S)
% Estimate the influence of each bit in S
% Inputs:
%   fittingfn - 
%   nbits - 
%   restricted_bits - index of bits restricted to 0
%   restriction - vector of values for restricted bits
%   nsamples - number of function queries to be made (need to be even number)
%   cprob - prob of sampling a 1 in generating bit vectors
%   S - set of bits for which the degree-1 fourier coefs are calculated
%
% Outputs:
%   fc: Fourier coefficents


    if ~exist('cprob','var')
        cprob = 0.5;
    end
    if ~exist('S','var')
        S = 1:nbits;
    end
    
    fc = zeros(1,length(S));
    X = zeros(nsamples, nbits);
    F = zeros(nsamples, 1);
    
    non_restricted_bits = setdiff(1:nbits, restricted_bits);
    
    % Generate half of the samples by random sampling
    for i = 1:2:nsamples
        x = zeros(1,nbits);
        x(non_restricted_bits) = randsrc(1, nbits-length(restricted_bits), [0, 1; 1-cprob, cprob]);     % random sample x
        F(i) = feval(fittingfn, x);     % evaluate function value
        X(i,:) = x;
    end
    
    % generate the other half by flipping relevent bits
    for j = 1:length(S)
        tbit = S(j);
            for i = 1:2:nsamples
                X(i+1, :) = X(i, :);
                X(i+1, tbit) = ~X(i+1, tbit);
                
%                 if F(i) == 1 && X(i, tbit) == 0
%                     F(i+1) = 1;
%                 elseif F(i) == 0 && X(i, tbit) == 1
%                     F(i+1) = 0;
%                 else
%                     F(i+1) = feval(fittingfn, X(i+1, :));     % evaluate function value
%                 end
                
                if F(i) == -1 && X(i, tbit) == 0
                    F(i+1) = -1;
                elseif F(i) == 1 && X(i, tbit) == 1
                    F(i+1) = 1;
                else
                    F(i+1) = feval(fittingfn, X(i+1, :));     % evaluate function value
                end
                
            end
            fc(j) = estimate_fc_OD1(X, F, tbit, cprob);
    end

end

function fc_s = estimate_fc_OD1(X, F, S, pb)
% S is the index of a single point
    
    Chi_q_mu_q = ((pb-1)/pb).^X(:,S).*(pb/(1-pb)).^sum(X,2);
%     fc_s = sum(F.*Chi_q_mu_q);
    fc_s = mean(F.*Chi_q_mu_q);
    % fc should be -(1-pb)^(ndata-1)*fc, but we can drop the constant
    % coefficients (1-pb)^(ndata-1) while using the rank info of fc
    %%
%     fc_s = -fc_s;
    
%     parity = -(-1).^X(:,S);
%     fc_s = 1* mean(F.*parity);
    
end