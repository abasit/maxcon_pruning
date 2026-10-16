function influence_jobs(jobfile)
% One MATLAB session over a job list written by maxcon_pruning (synthetic or real run, do_wi_mbf): WI (Zhang et
% al., TPAMI 2022) and MBF-MaxCon (Tennakoon et al., CVPR 2021), the authors' code unmodified
% (external/WI-MaxCon/functions, external/MBF-MaxCon/maxcon_BMF and utils), at each author's own demo settings: 100
% samples per round, local expansion on, rho = (p+1)/n for WI (demo_WI_linearfitting1.m) and (p+3)/n for MBF-MaxCon
% with 'maxcon-Linf' (maxcon_linear_demo.m), p the minimal sample size. A job is one repetition of both methods on
% one instance, each with rng(seed) of its own. The loops only see a fitting function (feasible or not) and a
% minimax function (the basis of the Chebyshev fit of a set), by the job's problem:
%   linear      the demos' functions (MaxConMBF_simple_example.m and bool_fn.m), with the Chebyshev fit of Chin et al.
%               2015 that both call; both count a set as feasible when its value is < eps, where ours have <= eps
%   homography  their code has none, so the fitting function is the pruning's L-infinity linear program (feasible
%               unless linprog proves it infeasible), and the minimax function returns the matches with nonzero duals
%               in the same program with t free, at least p + 1 of them
% Appends one JSON line per job and method to the list's results file: name, method, rep, seed, q, size, returned and
% removed (0-based indices into the instance), n_removed, fevals (MBF-MaxCon only) and seconds, or error.
% Usage: matlab -singleCompThread -batch "addpath('matlab'); influence_jobs('results/jobs/influence_authors.json')"
here = fileparts(mfilename('fullpath'));
ext = fullfile(here, '..', 'external');
addpath(fullfile(ext, 'MBF-MaxCon', 'maxcon_BMF'));
addpath(fullfile(ext, 'MBF-MaxCon', 'utils'));
addpath(fullfile(ext, 'WI-MaxCon', 'functions'));
warning('off', 'all');
L = jsondecode(fileread(jobfile));
jobs = L.jobs; if ~iscell(jobs), jobs = num2cell(jobs); end
fprintf('%s: %d jobs\n', jobfile, numel(jobs));
for i = 1:numel(jobs)
    J = jobs{i}; S = load(J.file); th = J.eps;
    if strcmp(J.problem, 'linear')
        Data = [S.A, S.b(:)]; nbits = size(Data, 1); p = size(S.A, 2);
        fit_mbf = @(x) bool_mbf(x, Data, th, nbits, p); mm_mbf = @(x) solve_minmax(x, Data);
        fit_wi = @(x) bool_fn(x, Data, th, p); mm_wi = @(x) minmax_linearsolver1(Data, x);
    else
        nbits = size(S.x1, 2); p = 4;
        fit_mbf = @(x) bool_lp(x, S.x1, S.x2, th, p); mm_mbf = @(x) basis_lp(x, S.x1, S.x2, th, p);
        fit_wi = fit_mbf; mm_wi = mm_mbf;
    end
    for method = {'MBF', 'WI'}
        m = method{1};
        r = struct('name', J.name, 'method', m, 'rep', J.rep);
        try
            if strcmp(m, 'MBF')
                r.seed = J.seed_mbf; rng(r.seed); r.q = (p + 3) / nbits; t0 = tic;
                [fixed, fevals] = maxcon_BMF('maxcon-Linf', fit_mbf, nbits, mm_mbf, 100, r.q, true);
                r.seconds = toc(t0); r.fevals = fevals;
            else
                r.seed = J.seed_wi; rng(r.seed); r.q = (p + 1) / nbits; t0 = tic;
                fixed = maxcon_WI(fit_wi, mm_wi, nbits, 100, r.q, true);
                r.seconds = toc(t0);
            end
            fixed = sort(fixed(:))';
            r.removed = num2cell(fixed - 1); r.n_removed = numel(fixed);        % num2cell: a list even of one
            r.returned = num2cell(setdiff(1:nbits, fixed) - 1); r.size = numel(r.returned);
            fprintf('%s %s rep %d: returned %d, removed %d, %.2f s\n', J.name, m, J.rep, r.size, r.n_removed, ...
                r.seconds);
        catch e
            r.error = e.message; fprintf('%s %s rep %d: ERROR %s\n', J.name, m, J.rep, e.message);
        end
        fid = fopen(L.out, 'a'); fprintf(fid, '%s\n', jsonencode(r)); fclose(fid);
    end
end
end

function f = bool_mbf(x, data, th, nbits, d)
% The fitting function of MaxConMBF_simple_example.m (1 feasible, -1 infeasible).
    x = x > 0.5;
    if sum(x) < d + 1
        f = int8(1);
    else
        f = int8(myFitTchebycheff_dist(data(x, :)') < th) * 2 - int8(1);
    end
end

function sol = solve_minmax(x, data)
% The minimax function of MaxConMBF_simple_example.m.
    [~, dd, basis] = myFitTchebycheff(data(x, :)');
    sol.basis = basis;
    sol.d = dd;
end

function f = bool_lp(x, x1, x2, th, p)
% The fitting function on a homography instance: 1 for a feasible set, -1 for one linprog proves infeasible. A set of
% at most p matches is feasible without a program, and any other failure of the program counts as feasible.
    S = find(x > 0.5);
    if numel(S) <= p
        f = int8(1);
    else
        f = int8(lp_homography(x1, x2, S, th, true) ~= 2) * 2 - int8(1);
    end
end

function sol = basis_lp(S, x1, x2, th, p)
% The minimax function on a homography instance: the positions in S of the matches with nonzero duals in the program
% with t free, at least p + 1 of them (the first p + 1 positions when the program is not solved).
    k = numel(S);
    [st, lam] = lp_homography(x1, x2, S, th, false);
    if st ~= 0 || isempty(lam)
        sol.basis = 1:min(k, p + 1); return;
    end
    w = max(reshape(abs(lam), k, 4), [], 2);                 % the largest dual of each match's four rows
    [~, order] = sort(-w);
    sol.basis = sort(order(1:min(max(p + 1, nnz(w > 1e-9)), k)))';
end

function [st, lam] = lp_homography(x1, x2, S, th, bounded)
% The pruning's program on the matches S (problems/homography.py, oracle_lp): min t over H (all nine entries) and t,
% |(h1 - x' h3) . x| <= th w + t and |(h2 - y' h3) . x| <= th w + t, the depths w = h3 . x summing to |S|, in Hartley
% frames; t <= 0 when bounded. st is 0 solved, 2 proven infeasible, 3 otherwise; lam the duals of the 4|S| inequality
% rows (row r belongs to match mod(r - 1, |S|) + 1).
    k = numel(S);
    X1 = x1(:, S) ./ x1(3, S); X2 = x2(:, S) ./ x2(3, S);
    T1 = hartley_T(X1); T2 = hartley_T(X2); X1 = T1 * X1; X2 = T2 * X2; e = th * T2(1, 1);
    x = X1(1, :)'; y = X1(2, :)'; xp = X2(1, :)'; yp = X2(2, :)'; o = ones(k, 1); z = zeros(k, 1);
    A = zeros(0, 10);
    for sgn = [1, -1]
        c = sgn * xp + e; A = [A; sgn * x, sgn * y, sgn * o, z, z, z, -c .* x, -c .* y, -c, -o]; %#ok<AGROW>
        c = sgn * yp + e; A = [A; z, z, z, sgn * x, sgn * y, sgn * o, -c .* x, -c .* y, -c, -o]; %#ok<AGROW>
    end
    Aeq = [0, 0, 0, 0, 0, 0, sum(x), sum(y), k, 0];
    lb = -inf(10, 1); ub = inf(10, 1); if bounded, ub(10) = 0; end
    [~, ~, flag, ~, lambda] = linprog([zeros(9, 1); 1], A, zeros(4 * k, 1), Aeq, k, lb, ub, ...
        optimoptions('linprog', 'Display', 'off'));
    if flag == 1
        st = 0; lam = lambda.ineqlin;
    elseif flag == -2
        st = 2; lam = [];
    else
        st = 3; lam = [];
    end
end

function T = hartley_T(x)
% Hartley normalisation of 3 x k points (last row 1): centroid 0, mean distance sqrt 2.
    c = mean(x(1:2, :), 2); d = mean(sqrt(sum((x(1:2, :) - c).^2, 1)));
    if d > 0, s = sqrt(2) / d; else, s = 1; end
    T = [s, 0, -s * c(1); 0, s, -s * c(2); 0, 0, 1];
end
