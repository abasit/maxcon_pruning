
% function [theta, Inls, runtime] = linearFit_homo2(x, y, th, method, theta0, numTrials,cut_t)
function [theta, Inls, runtime] = linearFit_homo2(x, y, th, method, numTrials,cut_t)
    
    nbits = size(x,1);
    d = size(x,2);
    Data = [x,y];
    
    if strcmp(method, 'RANSAC')   %% RANSAC 
        t1=tic;
%         [theta] = linearFitRANSAC1(x, y, th);   
        [theta, Inls, ~] = runRANSAC(Data', th, nbits, d, numTrials, 0, 0, cut_t);
        runtime = toc(t1);        
        
    elseif strcmp(method, 'LORANSAC')    %% Lo-RANSAC
        t2=tic;
%         [theta] = linearFitLORANSAC1(x, y, th); 
        [theta, Inls, ~] = runLoRANSAC(Data', th, nbits, d, numTrials, 0, 0, cut_t);
        runtime = toc(t2);         
    end
    
    %% Find Inliers
%     if (isempty(theta))
%         theta = theta0;
%     end
%     ss = sum(reshape(abs(y-x*theta)<=th,2,nbits/2));
%     Inls = find(ss==2);
end