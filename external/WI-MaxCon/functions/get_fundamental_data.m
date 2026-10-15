function [data, th] = get_fundamental_data(data_file)

load(data_file);
th = 0.01; 

% data.im1 = double(img1);
% data.im2 = double(img2);
% orig_data = data;
X1 = data(1:3,:);
X2 = data(4:6,:);
[X1_norm, ~] = normalise2dpts(X1);
[X2_norm, ~] = normalise2dpts(X2);
            
data = [X1_norm;X2_norm];

end