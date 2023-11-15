/*!@file Apps/BorderWatch/BorderWatchData.C */

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
// Primary maintainer for this file: Laurent Itti
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Apps/BorderWatch/BorderWatchData.C $
// $Id: BorderWatchData.C 13059 2010-03-26 08:14:32Z itti $
//

#include "Apps/BorderWatch/BorderWatchData.H"

#include <iostream>
#include <cstdio>

// ######################################################################
std::ostream& operator<<(std::ostream& os, const BorderWatchData& bwd)
{
  os<<"of="<<bwd.oframe<<" if="<<bwd.iframe<<" score="<<bwd.score<<" x="<<bwd.salpoint.i<<
    " y="<<bwd.salpoint.j<<" sal="<<bwd.saliency<<" ener="<<bwd.energy<<" uniq="<<bwd.uniqueness<<
    " entr="<<bwd.entropy<<" rand="<<bwd.rand<<" kls="<<bwd.KLsurprise<<" msds="<<bwd.MSDsurprise<<bwd.itime;
  return os;
}

// ######################################################################
std::istream& operator>>(std::istream& is, BorderWatchData& bwd)
{
  std::string line; getline(is, line); char d1[10], d2[10], d3[10], d4[10], d5[10], d[30];
  sscanf(line.c_str(), "of=%i if=%i score=%e x=%d y=%d sal=%e ener=%e uniq=%e entr=%e rand=%e kls=%e "
         "msds=%e %s %s %s %s %s\n",
         &bwd.oframe, &bwd.iframe, &bwd.score, &bwd.salpoint.i, &bwd.salpoint.j, &bwd.saliency,
         &bwd.energy, &bwd.uniqueness, &bwd.entropy, &bwd.rand, &bwd.KLsurprise, &bwd.MSDsurprise,
         d1, d2, d3, d4, d5);
  sprintf(d, "%s %s %s %s %s", d1, d2, d3, d4, d5); bwd.itime = d;
  return is;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
