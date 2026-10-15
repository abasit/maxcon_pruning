function [A, b] = genMatrix_Homography_linear(X1, X2)

    nbpoints = size(X1,2);
    AA = [];
    bb = [];   
    for k=1:nbpoints
        x1 = X1(1,k); 
        y1 = X1(2,k);
        x2 = X2(1,k); 
        y2 = X2(2,k);
        
        ai1 = [x1,y1,1,0,0,0,-x1*x2,-y1*x2]; 
        ai2 = [0,0,0,x1, y1, 1,-x1*y2,-y1*y2]; 
        AA = [AA; ai1; ai2];
        bb = [bb; [x2; y2]];         
    end    
    A = AA;
    b = bb;    
end