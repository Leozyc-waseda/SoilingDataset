% Saliency
%
% This program is part of the iLab Neuromorphic Vision C++ Toolkit.
% For the Toolkit contact Laurent Itti <itti@usc.edu>.
% For this MEX verion of the program contact Dirk Walther <walther@caltech.edu>
% April 2003
%
% Saliency computes which spots in an image are salient.
%
% usage: 
% [num_sal, coords, times, salmap, modfunc, areas, labels] = Saliency(image, ...
% targets, max_time, foa_size, weights, normtype, smoothMethod, levels);
%
% The return values are:
%     num_sal    the number of salient spots found
%     coords     (2, num_sal) array that contains the x and y coordinates
%                of the salient spots in (1,:) and (2,:), respectively 
%     times      (1, num_sal) array that contains the volution times
%     salmap     the saliency map (normalized between 0 and 1)
%     modfunc    (num_sal, height, width) array that contains the modulation
%                functions for the salient spots, maxnormalized to 1.0
%     areas      (1, num_sal) array that contains the number of pixels for
%                salient spot that have contributed to the modulation function
%     labels     {num_sal} cell array containing the label strings from the 
%                size info analysis
%
% The only required argument is:
%     image      the input image
%
% Optional paramters that have pre-assigned default values are:
%     targets    map of the same size as the input image, in which
%                targets for the focus of attention are 255 and the 
%                rest is zero. Pass a scalar 0 (the default) if you have 
%                no targets to look for. 
%     max_time   the maximum amount of (simulated) time that the 
%                saliency map should evolve in seconds (default: 0.7)
%     foa_size   the size of the focus of attention in pixels. The 
%                default is 1/12 of min(height, width) of the input 
%                image. Pass -1 if you want to use the default.
%     weights    Vector of length 3 containing the weights for the 
%                following channels (in this order): 
%                [wIntens, wOrient, wColor]
%                (default: [1.0 1.0 1.0])
%     normtype   normalization type; see fancynorm.H 
%                (default: 2 = VCXNORM_FANCY)
%  smoothMethod  method used for smoothing the shapeEstimator masks; 
%                see ShapeEstimatorModes.H; (default: 1 = Gaussian Smoothing)
%     levels     Vector of length 6 containing the following parameters:
%                [sm_level, level_min, level_max, delta_min, delta_max, nborients]
%                (default: [4 2 4 3 4 4])
%
