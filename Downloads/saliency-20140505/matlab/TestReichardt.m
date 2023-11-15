% [avg, min, max, levMap] = 
%   TestReichardt(img_seq, [dx dy], depth, lev)
%
% test the ReichardtPyramid (get tuning curves).
%
% Input Arguments:
%   img_seq   a 3d array, containing a series of images
%   [dx dy]   displacement determining the direction and magnitude of 
%             motion to be tested
%   depth     depth of the ReichardtPyramid
%   lev       level for which the Reichardt maps are returned
%
% Return Values:
%   avg       2d array containg the avg activation of each level of the 
%             ReichardtPyramid for each input image
%   min       the same, only this time it's the minimum
%   max       the same with the maximum
%   levMap    3d array containing the 'lev'th Reichardt map for each frame
%
