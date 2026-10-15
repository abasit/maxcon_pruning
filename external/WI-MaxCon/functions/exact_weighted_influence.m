% exact_weighted_influence.m
function ext_wi = exact_weighted_influence(fittingfn, nbits, q, d)
% Input: fittingfn: fitting function
%        nbits: number of data points
%        q: probability used in weighted influence
%        d: size of minimal subset
% Output: ext_wi: exact value of weighted influence

% if ~exist('q','var')
%     q = 0.5;
% end

q1 = (q-1)/q;
q2 = q/(1-q);

ext_wi = zeros(1,nbits);

for i_level = nbits:-1:d+1
    x = zeros(1,nbits);
    candidates = nchoosek(1:nbits,i_level);
    for j = 1:nchoosek(nbits,i_level)
%         fprintf('Iteration: (%d,%d)\n',i_level,j);
        x(candidates(j,:)) = 1;
        eva_f = feval(fittingfn,x);
        if eva_f == 1
            ext_wi = ext_wi-q2^i_level.*q1.^x;
        end
        x = zeros(1,nbits);
    end
end

% normalize the exact weighted influence
% ext_wi = ext_wi./max(ext_wi);

end