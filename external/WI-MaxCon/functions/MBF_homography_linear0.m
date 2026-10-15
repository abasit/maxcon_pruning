% MBF_homography_linear.m
function [theta, inliers] = MBF_homography_linear(A, y, th, theta0, nsamples, q, use_local)
% linearised homography with linear residual

Data = [A,y];
nbits = size(Data,1);
d = 8;
% nsamples = 100;     % number of samples for fourier estimation

fittingfn = @(x) bool_function_line_concensus(x, Data, th, nbits, d); % 1: feasible, -1: infeasible

minmaxfn = @(x) minmax_linearsolver1(Data,x);

% Run algorithm maxcon-Linf with local update
[fixed_bits, ~] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, nsamples, q, use_local);
inliers_idx  = setdiff(1:nbits, fixed_bits);
sol = feval(minmaxfn,inliers_idx);
theta = sol.theta;
if isempty(theta) % in case there are some issues
    theta = theta0;
    disp('Problem here');
end
ss = sum(reshape(abs(y-A*theta)<=th,2,nbits/2));
inliers = find(ss==2);

end
