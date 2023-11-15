/*!@file AppPsycho/make-gabor-snake.C */

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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/make-gabor-snake.C $
// $Id: make-gabor-snake.C 9083 2007-12-12 17:59:53Z rjpeters $
//

#ifndef APPPSYCHO_MAKE_GABOR_SNAKE_C_DEFINED
#define APPPSYCHO_MAKE_GABOR_SNAKE_C_DEFINED

#include "Psycho/SearchArray.H"
#include "Psycho/GaborPatch.H"
#include "Psycho/GaborSnake.H"
#include "Raster/Raster.H"

int main()
{
  GaborPatchItemFactory f(0, 0, 0, 15.0, 7.5);

  GaborSnake snake(24 /* itsForegNumber */,
                   90.0 /* itsForegSpacing */,
                   0 /* itsForegSeed */,
                   0 /* itsForegPosX */,
                   0 /* itsForegPosY */,
                   f);

  SearchArray g(Dims(1536, 1024), 96.0, 72.0);

  // pull in elements from the snake
  for (size_t n = 0; n < snake.numElements(); ++n)
    g.addElement(snake.getElement(n));

  g.generateBackground(f);

  Raster::WriteGray(g.getImage(), "foo.png");
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPPSYCHO_MAKE_GABOR_SNAKE_C_DEFINED
