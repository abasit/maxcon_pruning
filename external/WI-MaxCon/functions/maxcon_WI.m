
function fixed_bits = maxcon_WI(fittingfn, minmaxfn, nbits, nsamples, q, use_local)
    % fittingfn : Quary access to BMF
    % nbits: Number of input bits
    % minmaxfn: L-inf function access
    % nsamples: Number of samples to calculate fourier coeff
    % q: Positive Probability P(x_1 = 1)
    % use_local: use the local expantion step if = 1
    % fixed_bits: index set of outliers

    all_bits = 1:nbits;
    fixed_bits = [];

    for i = 1:nbits

        remaining_bits = setdiff(all_bits, fixed_bits);

        sol = feval(minmaxfn, remaining_bits);
        basis = sol.basis;
        basis = remaining_bits(basis);
        
        % Estimate deg-1 coeficents for basis datapoints of the restricted
        % function  
        %% used in KITTI
        nsamples = min(nsamples,floor(0.5*length(remaining_bits)));
        
        fc = estimate_weighted_influences(fittingfn, nbits, fixed_bits, nsamples, q, basis);
        
        % Remove the basis point with max influence from the dataset 
        [~, rbit] = max(fc);
        rbit = basis(rbit);
        fixed_bits = [fixed_bits, rbit];

        % check for feasiability
        x = ones(1, nbits);
        x(fixed_bits) = 0;
        f_value = feval(fittingfn, x);  % evaluate function value
%         if f_value == 0 % feasible
%             break;
%         end
        %%
        if f_value == 1 % feasible
            break;
        end

    end

    % perform local expansion
    if use_local
        remaining_bits = setdiff(all_bits, fixed_bits);
        x = zeros(1, nbits);
        x(remaining_bits) = 1;

        is_sol = 1;
        i_hold = length(remaining_bits);
        while is_sol==1
            [is_sol, x, i_hold] = local_expand1(fittingfn, x, [remaining_bits, fixed_bits], i_hold);
        end
        
        fixed_bits = find(x==0);

    end

end



function [is_sol, alpha_n, i_hold] = local_expand1(fittingfn, alpha_0, all_bits, i_hold)
    is_sol = 0;
    for i = i_hold+1:length(all_bits)
        alpha_n = alpha_0;
        alpha_n(all_bits(i)) = 1;
        f_value = feval(fittingfn, alpha_n);
%         if f_value == 0    %feasible
%             is_sol = 1;
%             i_hold = i;
%             return;
%         end
        if f_value == 1    %feasible
            is_sol = 1;
            i_hold = i;
            return;
        end
    end
    alpha_n = alpha_0;
end