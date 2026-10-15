function astar_jobs(jobfile)
% One MATLAB session over a job list written by maxcon_pruning.astar (the synthetic experiment): A*-NAPA-DIBP, the
% authors' code unmodified (external/MaxConTreeSearch, linearFitV2), with SeDuMi 1.3.4 from external/MBF-MaxCon, on one
% thread (start MATLAB with -singleCompThread). One untimed warm-up call on the list's warm-up instance (limit 30 s),
% then the jobs in order, job j on rows keep + 1 of its instance with its limit cap. Appends one JSON line per job to
% the list's results file: name, n, limit, set (the consensus set, 0-based indices into the full instance), consensus,
% time (linearFitV2's clock), hit_limit, nodes, nodibp, h_init, ub_init, level_max and sol, or error if the solver
% throws.
% Usage: matlab -singleCompThread -batch "addpath('matlab'); astar_jobs('results/jobs/surv_d8_t50.json')"
here = fileparts(mfilename('fullpath'));
ext = fullfile(here, '..', 'external');
addpath(genpath(fullfile(ext, 'MaxConTreeSearch', 'solvers')));
rmpath(genpath(fullfile(ext, 'MaxConTreeSearch', 'solvers', 'sedumi')));          % SeDuMi 1.21, not built
sd = fullfile(ext, 'MBF-MaxCon', 'linearASTAR', 'sedumi-master'); addpath(sd, fullfile(sd, 'conversion'));   % 1.3.4
warning('off', 'all');
L = jsondecode(fileread(jobfile));
W = load(L.warmup);
evalc('linearFitV2(W.A, W.b(:), W.x0(:), W.epsilon, ''A*-NAPA-DIBP'', 30);');   % warm-up, untimed
fprintf('%s: %d jobs; threads %d\n', jobfile, numel(L.jobs), maxNumCompThreads);
for i = 1:numel(L.jobs)
    J = L.jobs(i); limit = J.cap;
    S = load(J.file); keep = J.keep(:)' + 1; A = S.A(keep, :); b = S.b(keep); b = b(:);
    r = struct('name', J.name, 'n', numel(b), 'limit', limit);
    try
        txt = evalc('[sol, outl, nodes, hInit, ubInit, levelMax, nodibp, secs] = linearFitV2(A, b, S.x0(:), S.epsilon, ''A*-NAPA-DIBP'', limit);');
        inl = setdiff(1:numel(b), outl);
        r.set = keep(inl) - 1; r.consensus = numel(inl); r.time = secs; r.hit_limit = contains(txt, 'reaching time limit');
        r.nodes = nodes; r.nodibp = nodibp; r.h_init = hInit; r.ub_init = ubInit; r.level_max = levelMax; r.sol = sol(:)';
        fprintf('%s: consensus %d of %d, %.2f s, limit hit %d, nodes %d\n', J.name, r.consensus, r.n, secs, ...
            r.hit_limit, nodes);
    catch e
        r.error = e.message; fprintf('%s: ERROR %s\n', J.name, e.message);
    end
    fid = fopen(L.out, 'a'); fprintf(fid, '%s\n', jsonencode(r)); fclose(fid);
end
end
