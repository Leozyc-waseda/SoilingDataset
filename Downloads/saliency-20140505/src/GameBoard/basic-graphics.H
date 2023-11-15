/*!@file GameBoard/basic-graphics.H some utilities for displaying stimuli*/
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
//written by Nader Noori, April 2008
#ifndef GAMEBOARD_BASICGRAPHICS_H_DEFINED
#define GAMEBOARD_BASICGRAPHICS_H_DEFINED

#include <SDL/SDL.h>
#include <SDL/SDL_image.h>


SDL_Surface *load_image( std::string filename );

SDL_Surface *getABlankSurface(int w ,  int h);

void dumpSurface(SDL_Surface* surface);

void apply_surface( int x, int y, SDL_Surface& source, SDL_Surface& destination , SDL_Rect& clip ) ;

void fillRectangle(SDL_Surface* surface , const Uint32 pc , uint offX  , uint offY  , const uint w , const uint h ) ;

void fillOval(SDL_Surface* surface , const Uint32 pc , int offX  , uint offY  , const int w , const int h , const Uint32 bgpc);

void fillLinearRadiant(SDL_Surface* surface , const Uint32 pc , uint offX , uint offY, const int R) ;

void fillQuadricRadiant(SDL_Surface* surface , const Uint32 pc , uint offX , uint offY, const int R) ;

void fillCubicRadiant(SDL_Surface* surface , const Uint32 pc , uint offX , uint offY, const int R) ;

void drawRectangle(SDL_Surface* surface , const Uint32 pc , uint offX  , uint offY  , const uint w , const uint h , const uint f=1);

void drawCircle(SDL_Surface* surface , const Uint32 pc , uint offX  , uint offY  , const uint r, const uint d=1);

void drawArc(SDL_Surface* surface , const Uint32 pc , uint offX  , uint offY  , const uint r , const uint d , float teta1, float teta2 , float deltaAlpha=0.1f);

void drawRectangleFromImage(SDL_Surface* surface , SDL_Surface* patch , uint offX  , uint offY  , const uint w , const uint h , const uint f=1);

Uint32 get_pixel32( SDL_Surface *surface, int x, int y ) ;

void put_pixel32( SDL_Surface *surface, int x, int y, Uint32 pixel ) ;

inline void putpixel(SDL_Surface* surface, int x, int y, Uint32 pixel) ;

void drawColorWheel(SDL_Surface* surface ,const std::vector<Uint32> colors,  int cX  , uint cY  , const int minRadius , const int maxRadius , const Uint32 bgpc);

#endif

