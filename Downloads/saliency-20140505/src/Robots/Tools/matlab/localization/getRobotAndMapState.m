%/*!@file Robots/matlab/localization/getRobotAndMapState.m  */
%
%
%// //////////////////////////////////////////////////////////////////// //
%// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
%// University of Southern California (USC) and the iLab at USC.         //
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
%// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
%// $HeadURL $
%// $Id $
%//

function [State, Map] = getRobotAndMapState(Particles)

   %find the most maximum particle
	maxP = 1;
	maxVal = Particles{maxP}.prob;

	for i=2:size(Particles, 2)
		if (Particles{i}.prob > maxVal)
			maxP = i;
			maxVal = Particles{maxP}.prob;
		end
	end

  State = Particles{maxP}.pos;
	Map = cell2mat(Particles{maxP}.landmarksPos(:));