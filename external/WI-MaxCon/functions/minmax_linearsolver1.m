function sol = minmax_linearsolver1(data,x)

%% minmax solver: fminmax
%     sol = minmax_linearsolver(data(x,:));
    
%% minmax solver: Chebycheff fitting
    [theta, s, basis ] = myFitTchebycheff(data(x, :)');
    sol.theta = theta;
    sol.basis = basis;
    sol.s = s;
    
%     [theta, s, basis ] = myFitTchebycheff_dist(data(x, :)');
%     sol.theta = theta;
%     sol.s = s;
%     sol.basis = basis;
    
end