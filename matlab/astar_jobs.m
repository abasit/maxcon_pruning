function astar_jobs(jobfile)
% One MATLAB session over a job list written by maxcon_pruning.astar: A*-NAPA-DIBP, the authors' code unmodified
% (external/MaxConTreeSearch), with SeDuMi 1.3.4 from external/MBF-MaxCon, on one thread (start MATLAB with
% -singleCompThread). Job j runs on the points keep + 1 of its instance file with its limit cap, by its problem:
%   linear      linearFitV2 on those rows of A and b, threshold eps, from the instance's random start x0
%   homography  pseudoConvFit on those matches, Hartley-normalised (x1 by T1, x2 by T2), in genMatrixHomography's
%               form, threshold eps T2(1,1), so that the L2 transfer error is at most eps pixels; from the job's init
%               (H in pixels, row-major) mapped into that frame and scaled to h33 = 1
% An untimed warm-up first, on the first job's first min(40, n) points with the limit 10 s. Appends one JSON line per
% job to the list's results file: name, n, limit, set (the consensus set, 0-based indices into the full instance),
% consensus, time (the solver's clock), hit_limit, nodes, nodibp, h_init, ub_init, level_max and model (theta, or H in
% pixels row-major), or error if the solver throws.
% Usage: matlab -singleCompThread -batch "addpath('matlab'); astar_jobs('results/jobs/surv_d8_t50.json')"
here = fileparts(mfilename('fullpath'));
ext = fullfile(here, '..', 'external');
addpath(genpath(fullfile(ext, 'MaxConTreeSearch', 'solvers')));
rmpath(genpath(fullfile(ext, 'MaxConTreeSearch', 'solvers', 'sedumi')));          % SeDuMi 1.21, not built
sd = fullfile(ext, 'MBF-MaxCon', 'linearASTAR', 'sedumi-master'); addpath(sd, fullfile(sd, 'conversion'));   % 1.3.4
warning('off', 'all');
L = jsondecode(fileread(jobfile));
jobs = L.jobs; if ~iscell(jobs), jobs = num2cell(jobs); end
J = jobs{1}; keep = J.keep(:)' + 1;
try, solve(build(J, keep(1:min(40, end))), 10); catch, end                        % warm-up, untimed
fprintf('%s: %d jobs; threads %d\n', jobfile, numel(jobs), maxNumCompThreads);
for i = 1:numel(jobs)
    J = jobs{i}; keep = J.keep(:)' + 1;
    r = struct('name', J.name, 'n', numel(keep), 'limit', J.cap);
    try
        P = build(J, keep);
        [txt, sol, outl, nodes, hInit, ubInit, levelMax, nodibp, secs] = solve(P, J.cap);
        inl = setdiff(1:numel(keep), outl);
        r.set = keep(inl) - 1; r.consensus = numel(inl); r.time = secs; r.hit_limit = contains(txt, 'reaching time limit');
        r.nodes = nodes; r.nodibp = nodibp; r.h_init = hInit; r.ub_init = ubInit; r.level_max = levelMax;
        r.model = model(P, sol);
        fprintf('%s: consensus %d of %d, %.2f s, limit hit %d, nodes %d\n', J.name, r.consensus, r.n, secs, ...
            r.hit_limit, nodes);
    catch e
        r.error = e.message; fprintf('%s: ERROR %s\n', J.name, e.message);
    end
    fid = fopen(L.out, 'a'); fprintf(fid, '%s\n', jsonencode(r)); fclose(fid);
end
end

function P = build(J, keep)
% The solver's input for the points keep (1-based) of the job's instance (see the header).
S = load(J.file); P.problem = J.problem;
if strcmp(P.problem, 'linear')
    P.A = S.A(keep, :); b = S.b(keep); P.b = b(:); P.x0 = S.x0(:); P.th = J.eps;
else
    [x1n, P.T1] = normalise(S.x1(:, keep)); [x2n, P.T2] = normalise(S.x2(:, keep));
    [P.A, P.b, P.c, P.d] = genMatrixHomography(x1n, x2n);
    P.th = J.eps * P.T2(1, 1);
    Hn = P.T2 * reshape(J.init, 3, 3)' / P.T1; Hn = Hn / Hn(3, 3); hn = Hn';
    P.x0 = hn(1:8)';                                                              % its first 8 entries, row-major
end
end

function [txt, sol, outl, nodes, hInit, ubInit, levelMax, nodibp, secs] = solve(P, limit)
% A*-NAPA-DIBP on P with the given limit; txt is the solver's printed output.
if strcmp(P.problem, 'linear')
    txt = evalc('[sol, outl, nodes, hInit, ubInit, levelMax, nodibp, secs] = linearFitV2(P.A, P.b, P.x0, P.th, ''A*-NAPA-DIBP'', limit);');
else
    txt = evalc('[sol, outl, nodes, hInit, ubInit, levelMax, nodibp, secs] = pseudoConvFit(P.A, P.b, P.c, P.d, P.x0, P.th, ''A*-NAPA-DIBP'', limit);');
end
end

function m = model(P, sol)
% The solution in the instance's frame: theta, or H in pixels (row-major).
if strcmp(P.problem, 'linear')
    m = sol(:)';
else
    H = P.T2 \ [sol(1:3)'; sol(4:6)'; sol(7:8)', 1] * P.T1; m = reshape(H', 1, []);
end
end

function [xn, T] = normalise(x)
% Hartley normalisation of 3 x n homogeneous points: centroid at the origin, mean distance sqrt(2).
c = mean(x(1:2, :), 2);
s = sqrt(2) / mean(sqrt(sum((x(1:2, :) - c).^2, 1)));
T = [s 0 -s*c(1); 0 s -s*c(2); 0 0 1];
xn = T * x;
end
