function influence_jobs(jobfile)
% One MATLAB session over a job list written by maxcon_pruning.synthetic.run (do_influence_authors): WI (Zhang et al.,
% TPAMI 2022) and MBF-MaxCon (Tennakoon et al., CVPR 2021), the authors' code unmodified (external/WI-MaxCon/functions,
% external/MBF-MaxCon/maxcon_BMF and utils, with the Chebyshev fit of Chin et al. 2015 that both call), on the linear
% instances of the list at each author's own demo settings: 100 samples per round, local expansion on, rho = (d+1)/n
% for WI (demo_WI_linearfitting1.m) and (d+3)/n for MBF-MaxCon with 'maxcon-Linf' (maxcon_linear_demo.m). The fitting
% functions are those of the demos (MaxConMBF_simple_example.m and bool_fn.m); both count a set as feasible when its
% value is < eps, where ours have <= eps. rng(seed) per job and method. Appends one JSON line per job and method to the
% list's results file: name, method, q, size, returned and removed (0-based indices into the instance), n_removed,
% fevals (MBF-MaxCon only) and seconds, or error.
% Usage: matlab -singleCompThread -batch "addpath('matlab'); influence_jobs('results/jobs/influence_authors.json')"
here = fileparts(mfilename('fullpath'));
ext = fullfile(here, '..', 'external');
addpath(fullfile(ext, 'MBF-MaxCon', 'maxcon_BMF'));
addpath(fullfile(ext, 'MBF-MaxCon', 'utils'));
addpath(fullfile(ext, 'WI-MaxCon', 'functions'));
warning('off', 'all');
L = jsondecode(fileread(jobfile));
fprintf('%s: %d jobs\n', jobfile, numel(L.jobs));
for i = 1:numel(L.jobs)
    J = L.jobs(i);
    S = load(J.file);
    Data = [S.A, S.b(:)];
    nbits = size(Data, 1); d = size(S.A, 2); th = S.epsilon;
    for method = {'MBF', 'WI'}
        m = method{1};
        r = struct('name', J.name, 'method', m);
        try
            if strcmp(m, 'MBF')
                rng(J.seed_mbf);
                fittingfn = @(x) bool_mbf(x, Data, th, nbits, d);
                minmaxfn = @(x) solve_minmax(x, Data);
                r.q = (d + 3) / nbits;
                t0 = tic;
                [fixed, fevals] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, 100, r.q, true);
                r.seconds = toc(t0); r.fevals = fevals;
            else
                rng(J.seed_wi);
                fittingfn = @(x) bool_fn(x, Data, th, d);
                minmaxfn = @(x) minmax_linearsolver1(Data, x);
                r.q = (d + 1) / nbits;
                t0 = tic;
                fixed = maxcon_WI(fittingfn, minmaxfn, nbits, 100, r.q, true);
                r.seconds = toc(t0);
            end
            fixed = sort(fixed(:))';
            r.removed = fixed - 1; r.n_removed = numel(fixed);
            r.returned = setdiff(1:nbits, fixed) - 1; r.size = numel(r.returned);
            fprintf('%s %s: returned %d, removed %d, %.2f s\n', J.name, m, r.size, r.n_removed, r.seconds);
        catch e
            r.error = e.message; fprintf('%s %s: ERROR %s\n', J.name, m, e.message);
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
