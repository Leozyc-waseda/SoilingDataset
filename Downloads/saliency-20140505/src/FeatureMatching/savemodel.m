%/*!@file FeatureMatching/savemodel.m Save a model from the dpm matlab structure to a file, that can be read from the C prog


%// //////////////////////////////////////////////////////////////////// //
%// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
%// by the University of Southern California (USC) and the iLab at USC.  //
%// See http://iLab.usc.edu for information about this project.          //
%// //////////////////////////////////////////////////////////////////// //
%// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
%// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
%// in Visual Environments, and Applications'' by Christof Koch and      //
%// Laurent Itti, California Institute of Technology, 2001 (patent       //
%// pending; application number 09/912,225 filed July 23, 2001; see      //
%// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
%// //////////////////////////////////////////////////////////////////// //
%// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
%//                                                                      //
%// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
%// redistribute it and/or modify it under the terms of the GNU General  //
%// Public License as published by the Free Software Foundation; either  //
%// version 2 of the License, or (at your option) any later version.     //
%//                                                                      //
%// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
%// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
%// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
%// PURPOSE.  See the GNU General Public License for more details.       //
%//                                                                      //
%// You should have received a copy of the GNU General Public License    //
%// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
%// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
%// Boston, MA 02111-1307 USA.                                           //
%// //////////////////////////////////////////////////////////////////// //
%//
%// Primary maintainer for this file: Lior Elazary
%// $HeadURL$
%// $Id$
%//



function savemodel(modfile, model)

fid = fopen(modfile, 'wb');


numComponents = length(model.rules{model.start});
fwrite(fid, numComponents, 'int32');

for c = 1:numComponents
  rhs = model.rules{model.start}(c).rhs;
  layer = 1;
  root = -1;

  % assume the root filter is first on the rhs of the start rules
  if model.symbols(rhs(1)).type == 'T'
    % handle case where there's no deformation model for the root
    root = model.symbols(rhs(1)).filter;
  else
    % handle case where there is a deformation model for the root
    root = model.symbols(model.rules{rhs(1)}(layer).rhs).filter;
  end

  %Save the root filter
  %w = model.filters(root).w;
  w = model_get_block(model, model.filters(root));
  fwrite(fid, size(w), 'int32');
  for f = 1:size(w,3)
      d=w(:,:,f)'; %Transpose since we store in C as width,height
      fwrite(fid, d(:), 'double');
  end

  %Write the offset
  %offset = model.rules{model.start}(c).offset.w;
  offset = model_get_block(model, model.rules{model.start}(c).offset);
  fwrite(fid, offset, 'double');

  %Save the Parts
  %Write the number of parts
  fwrite(fid, length(rhs)-1, 'int32');
  for i = 2:length(rhs)
    %def = model.rules{rhs(i)}(layer).def.w;
    def = model_get_block(model, model.rules{rhs(i)}(layer).def);
    anchor = model.rules{model.start}(c).anchor{i};
    fi = model.symbols(model.rules{rhs(i)}(layer).rhs).filter;
    %w = model.filters(fi).w;
    w = model_get_block(model, model.filters(fi));

    fwrite(fid, anchor(1:3), 'double');
    fwrite(fid, def, 'double');

    fwrite(fid, size(w), 'int32');
    for f = 1:size(w,3)
      d=w(:,:,f)'; %Transpose since we store in C as width,height
      fwrite(fid, d(:), 'double');
    end
  end
end

fclose(fid);
