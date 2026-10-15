function SF_ab=topk_SpearmanDistance(list_a,list_b,topk)

[~,ord_a] = sort(list_a,'descend');
% [~,orig_ord_a] = sort(ord_a);
[~,ord_b] = sort(list_b,'descend');
% [~,orig_ord_b] = sort(ord_b);
list_a = ord_a;
list_b = ord_b;

Y = sort(unique([list_a(1:topk);list_b(1:topk)]));
[~,Ya] = ismember(Y,list_a(1:topk));
Ya(Ya==0) = topk+1;
[~,Yb] = ismember(Y,list_b(1:topk));
Yb(Yb==0) = topk+1;
SF_ab = norm(Ya-Yb,1);
% normalized SF
SF_ab = SF_ab/topk/(topk+1);
% SF_ab = SF_ab/topk;

end