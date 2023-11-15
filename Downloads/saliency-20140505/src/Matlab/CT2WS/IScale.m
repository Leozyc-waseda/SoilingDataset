% IScale - Non-linear scaling function for saliency color maps
% USAGE:
% s = Iscale(u)
% INPUT:
% u - a matrix representing the intensity channel image
% OUTPUT:
% s - a scaling matrix that is applied by point-wise multiplication to
%     the un-normalized RGBY color channel images.
function s = IScale(u)
    iThresh = 0.1*max(max(u)); % threshold at 10% of max intensity value
    ixSmall = find(u <= iThresh); % find below threshold pixels
    s = u;
    s(ixSmall) = 1; % Set below thresh points to an inocuous value.
    s = 1./s; % Invert element-wise
    s(ixSmall) = 0; % Set scaling for below thresh points to zero.
