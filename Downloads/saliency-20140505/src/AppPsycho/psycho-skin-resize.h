/*!@file AppPsycho/psycho-skin-indexing.C Psychophysics support for psycho-skin-indexing.h */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-skin-resize.h $
// $Id: psycho-skin-resize.h 10794 2009-02-08 06:21:09Z itti $
//

//resize.c - has all you need to get high-quality interpolated image scaling - up & down!
//please see resize.c for more information about who has contributed to this library

#ifndef __RESIZE_FILTERS__
#define __RESIZE_FILTERS__

#include <SDL/SDL.h>

//Here are the only 2 functions you need:
//NULL will be returned if the passed in surface "image" is invalid
SDL_Surface* SDL_ResizeFactor(SDL_Surface *image, float scalefactor,    int filter);
SDL_Surface* SDL_ResizeXY    (SDL_Surface *image, int new_w, int new_h, int filter);

//Here are overloaded C++ versions, with filter default as high quality.
#ifdef __cplusplus
SDL_Surface* SDL_Resize(SDL_Surface *image, float scalefactor,    int filter = 7);
SDL_Surface* SDL_Resize(SDL_Surface *image, int new_w, int new_h, int filter = 7);
#endif

/*The passed-in surface is freed by SDL_Resize, so it works nicely to pass in surfaces
  as themselves:
  e.g. pic = SDL_ResizeFactor(pic, 0.75, 7); (or pic = SDL_Resize(pic, 0.75);)
  This will shrink pic to 75% of original size. No other cleanup necessary.
  Another good way to use it is on initialization:
  e.g. SDL_Surface *pic = SDL_ResizeXY(SDL_LoadBMP("mypic.bmp"),50,50,7);
  This will give you mypic.png at size 50x50(regardless of original dimensions)
  if mypic.bmp did not load correctly, pic will be NULL.
*/

/*
Filters are as follows:
1 = box filter - fastest/ugliest.
2 =        triangle filter - possible visual anomalies
3 = bell filter - possible visual anomalies
4 = B_spline filter - here is where it starts to get good.
5 =        hermite filter - relatively fast, good quality
6 =        Mitchell filter - also speedy, good quality
7 = Lanczos3 filter - slowest, but by far best quality. Very sharp!
If filter is not specified, Lanczos3 will be selected by default
*/

/*The code should compile as either C or C++, without any fuss, except maybe you specifying
  which kind to compile it as... -Dave Olsen
*/

#endif

