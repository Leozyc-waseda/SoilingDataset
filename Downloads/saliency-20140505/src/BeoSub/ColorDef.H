/*!@file BeoSub/ColorDef.H */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/ColorDef.H $
// $Id: ColorDef.H 6182 2006-01-31 18:41:41Z rjpeters $
//

#ifndef BEOSUB_COLORDEF_H_DEFINED
#define BEOSUB_COLORDEF_H_DEFINED

std::vector<float> red(4,0.0F);
std::vector<float> green(4,0.0F);
std::vector<float> orange(4,0.0F);
std::vector<float> blue(4,0.0F);
std::vector<float> yellow(4,0.0F);
std::vector<float> white(4,0.0F);
std::vector<float> black(4,0.0F);
std::vector<float> brown(4,0.0F);


class ColorDef {

public:
ColorDef(){

//H2SV

//    red[0] = 0.70000F; red[1] = 0.94400F; red[2] = 0.67200F; red[3] = 0.96900F;
red[0] = 1.00000F; red[1] = 1.00000F; red[2] = 1.00000F; red[3] = 0.50000F;
//    green[0] = 0.53000F; green[1] = 0.36000F; green[2] = 0.44600F; green[3] = 0.92000F;
green[0] = 0.25000F; green[1] = 0.75000F; green[2] = 0.44600F; green[3] = 0.92000F;
//    orange[0] = 0.80000F; orange[1] = 0.80000F; orange[2] = 0.57600F; orange[3] = 0.96800F;
orange[0] = 0.75000F; orange[1] = 0.78000F; orange[2] = 0.57600F; orange[3] = 0.96800F;
//^^ not done yet
//    blue[0] = 0.00000F; blue[1] = 0.00000F; blue[2] = 0.44500F; blue[3] = 0.97200F;
blue[0] = 0.00000F; blue[1] = 0.00000F; blue[2] = 0.44500F; blue[3] = 0.97200F;
//    yellow[0]=1.00000F; yellow[1] = 0.62000F; yellow[2] = 0.54000F; yellow[3] = 0.97000F;
yellow[0]=0.91000F; yellow[1] = 0.63000F; yellow[2] = 1.00000F; yellow[3] = 1.00000F;
    white[0] = 0.70000F; white[1] = 0.70000F; white[2] = 0.04000F; white[3] = 1.00000F;
    black[0] = 0.60000F; black[1] = 0.66000F; black[2] = 0.24000F; black[3] = 0.70000F;
    brown[0] = 0.744817F; brown[1] = 0.882294F; brown[2] = 0.613244F; brown[3] = 0.624400F;

};

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // BEOSUB_COLORDEF_H_DEFINED
