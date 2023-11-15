/*!@file GameBoard/ColorDef.H */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Nader Noori(nnoori@usc.edu)
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/GameBoard/ColorDef.H $
//

#include "Image/PixelsTypes.H"


#ifndef GAMEBOARD_COLORDEF_H_DEFINED
#define GAMEBOARD_COLORDEF_H_DEFINED

PixRGB<byte> white ( 255,255,255 );
PixRGB<byte> red ( 255,0,0 );
PixRGB<byte> maroon (128,0,0);
PixRGB<byte> lime ( 0,255,0 );
PixRGB<byte> green (0,110,0);
PixRGB<byte> blue ( 0,0,255 );
PixRGB<byte> navy (0,0,110);
PixRGB<byte> black ( 0,0,0 );
PixRGB<byte> grey ( 50,50,50 );
PixRGB<byte> yellow (255,255,0);
PixRGB<byte> violet (128,0,128);
PixRGB<byte> orange (255,128,0);
PixRGB<byte> lightblue(85,221,255);
PixRGB<byte> pink(222,135,170);

class ColorDef {

public:
ColorDef(){

};

};


#endif // GAMEBOARD_COLORDEF_H_DEFINED
