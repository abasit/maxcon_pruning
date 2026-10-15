% WI_homography_linear.m
% function [theta, inliers,runtime] = WI_homography_linear(A, y, th, theta0, nsamples, q, use_local)
function [inliers,runtime] = WI_homography_linear(A, y, th, nsamples, q, use_local)
% linearised homography with linear residual

tic;
Data = [A,y];
nbits = size(Data,1);
d = 8;



% fittingfn1 = @(x) bool_function_line_concensus(x, Data, th, nbits, d); % 1: feasible, -1: infeasible
fittingfn1 = @(x) bool_fn(x, Data, th, d);
minmaxfn1 = @(x) minmax_linearsolver1(Data,x);

% Run algorithm maxcon-Linf with local update
fixed_bits = maxcon_WI(fittingfn1, minmaxfn1, nbits, nsamples, q, use_local);
% inliers_idx  = setdiff(1:nbits, fixed_bits);
% sol = feval(minmaxfn1,inliers_idx);
% theta = sol.theta;

inliers  = setdiff(1:nbits, fixed_bits);

%perform LS on the final consensus set
%compute initial inliers
% theta0 = theta;
% inls0 = find(abs(y-A*theta)<=th);
% theta = ((A(inls0,:)'*A(inls0,:))^-1)*A(inls0,:)'*(y(inls0));
% inliers = find(abs(y-A*theta)<=th);
% if (numel(inls0)>inliers)
% %     inliers = inls0;
%     theta = theta0;
% end


% if isempty(theta) % in case there are some issues
%     theta = theta0;
%     disp('Problem here');
% end
% ss = sum(reshape(abs(y-A*theta)<=th,2,nbits/2));
% inliers = find(ss==2);

runtime = toc;

end
