fid = fopen('fancytest.maxes', 'r'); gogo=fscanf(fid, '%s %f %f'); fclose(fid);
idx = 18:18:length(gogo); maxes = gogo(idx); maxes(1) = 1;

## #################################################################### ##
## The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the ##
## University of Southern California (USC) and the iLab at USC.         ##
## See http://iLab.usc.edu for information about this project.          ##
## #################################################################### ##
## Major portions of the iLab Neuromorphic Vision Toolkit are protected ##
## under the U.S. patent ``Computation of Intrinsic Perceptual Saliency ##
## in Visual Environments, and Applications'' by Christof Koch and      ##
## Laurent Itti, California Institute of Technology, 2001 (patent       ##
## pending; filed July 23, 2001, following provisional applications     ##
## No. 60/274,674 filed March 8, 2001 and 60/288,724 filed May 4, 2001).##
## #################################################################### ##
## This file is part of the iLab Neuromorphic Vision C++ Toolkit.       ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is free software; you can   ##
## redistribute it and/or modify it under the terms of the GNU General  ##
## Public License as published by the Free Software Foundation; either  ##
## version 2 of the License, or (at your option) any later version.     ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  ##
## that it will be useful, but WITHOUT ANY WARRANTY; without even the   ##
## implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      ##
## PURPOSE.  See the GNU General Public License for more details.       ##
##                                                                      ##
## You should have received a copy of the GNU General Public License    ##
## along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   ##
## to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   ##
## Boston, MA 02111-1307 USA.                                           ##
## #################################################################### ##
##
## Primary maintainer for this file: Laurent Itti <itti@usc.edu>
## $Id: fancyanalyze.m 6067 2005-12-20 19:08:27Z rjpeters $
##

nbline = 2; use = 0:2:((nbline*4)*2-1);

[X,map]=tiffread('fancytest000.tif'); I = ind2gray(X,map);
subplot(nbline,4,1); imshow(flipud(X),map);
[nr nc] = size(X);
hold off;

mm = max(maxes(use(1:nbline*4-1)+1));

for ii=1:(nbline*4-1)
 name=sprintf('fancytest%03d.tif', use(ii))
 [X,map]=tiffread(name); I = ind2gray(X,map);
 subplot(nbline,4,ii+1);
 mesh(I*maxes(use(ii)+1),ones(size(I)));
 axis([1 nr 1 nc 0 mm/1.1]); axis off;
 view(-37.5, 10);
 title(['Iteration ' int2str(use(ii))]);
end
