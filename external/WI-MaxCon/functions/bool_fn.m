%%%%%%%% Bool functions of interest %%%%%%%%%%%
function f = bool_fn(x, data, th, d)
    % f: {F_2}^n -> {0, 1}
    % Inputs: 
    % x:    input vector of nbits
    % Output:
    % 1: feasible, -1: infeasible
    
    x = x > 0.5;    % may only work for 0, 1 dictionary % turn double values as logical values
    if sum(x) < d + 1
%         f = int8(0);
        %%
        f = int8(1);
    else
        res = minmax_linearsolver1(data,x);
%         f = int8(1)-int8(res.s < th);
%         res = mbf_myFitTchebycheff_dist(data(x,:)');
%         f = int8(1)-int8(res < th);
        %%
        f = int8(res.s < th)*2-int8(1);
    end
end