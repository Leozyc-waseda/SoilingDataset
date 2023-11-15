/*!@file Foveator/test-Foveator.C Test the Foveator class */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Foveator/test-Foveator.C $
// $Id: test-Foveator.C 6191 2006-02-01 23:56:12Z rjpeters $
//

#include "Foveator/BlurFoveator.H"
#include "Foveator/Foveator.H"
#include "Foveator/LPTFoveator.H"
#include "Foveator/PyrFoveator.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"
#include "Util/log.H"

// ######################################################################
// ##### Main Program:
// ######################################################################

//! The main routine. Read images, process, output results.
int main (int argc, char **argv)
{
  if (argc != 2)
    {
      LFATAL("usage: test-Foveator file.[ppm|pgm]\n");
    }

  // read the image

  Image< PixRGB<byte> > img;

  img = Raster::ReadRGB( argv[1] );
  if (!img.initialized())
    {
      LFATAL("error loading image: %s", argv[1] );
    }

  // create output file names

  char *blr_out, *pyr_out, *lpt_out, *map_out;
  int filenameLength = strlen( argv[1] );
  blr_out = new char[filenameLength + 5];
  pyr_out = new char[filenameLength + 5];
  lpt_out = new char[filenameLength + 5];
  map_out = new char[filenameLength + 5];

  int lastdot = filenameLength - 1;
  while( argv[1][lastdot] != '.' )
  {
          lastdot--;
  }
  int i = 0;
  for( i = 0; i < lastdot; i++ )
  {
    blr_out[i] = argv[1][i];
    pyr_out[i] = argv[1][i];
    lpt_out[i] = argv[1][i];
    map_out[i] = argv[1][i];
  }

  blr_out[i++] = '_';
  blr_out[i++] = 'b';
  blr_out[i++] = 'l';
  blr_out[i++] = 'r';
  i -= 4;
  pyr_out[i++] = '_';
  pyr_out[i++] = 'p';
  pyr_out[i++] = 'y';
  pyr_out[i++] = 'r';
  i -= 4;
  lpt_out[i++] = '_';
  lpt_out[i++] = 'l';
  lpt_out[i++] = 'p';
  lpt_out[i++] = 't';
  i -= 4;
  map_out[i++] = '_';
  map_out[i++] = 'm';
  map_out[i++] = 'a';
  map_out[i++] = 'p';

  while( i < ( filenameLength + 4 ) )
  {
    blr_out[i] = argv[1][i-4];
    pyr_out[i] = argv[1][i-4];
    lpt_out[i] = argv[1][i-4];
    map_out[i] = argv[1][i-4];
    i++;
  }
  blr_out[i] = '\0';
  pyr_out[i] = '\0';
  lpt_out[i] = '\0';
  map_out[i] = '\0';

  // using Foveator classes

  BlurFoveator bf( img, 5 );
  PyrFoveator pf( img, 5 );
  pf.setBaseRect( 40, 40 );
  LPTFoveator lf( img, img.getWidth() / 2, img.getHeight() / 2 );

  // write foveated image to output files

  Image< PixRGB<byte> > blrImg;
  Timer tm;
  blrImg = bf.foveate();
  // blrImg = BlurFoveator::foveate( img, 5, 100, 100 );
  LINFO( "Time for BlurFoveation: %llums", tm.get() );
  Raster::WriteRGB( blrImg, blr_out );

  Image< PixRGB<byte> > pyrImg;
  tm.reset();
  pyrImg = pf.foveate();
  //  pyrImg = PyrFoveator::foveate( img, 5, 40, 40, 100, 100 );
  LINFO( "Time for PyrFoveation: %llums", tm.get() );
  Raster::WriteRGB( pyrImg, pyr_out );

  Image< PixRGB<byte> > lptImg;
  tm.reset();
  lptImg = lf.foveate();
  /*  lptImg = LPTFoveator::foveate( img, img.getWidth() / 2,
                                 img.getHeight() / 2, img.getWidth() / 2,
                                 img.getHeight() / 2, false ); */
  LINFO( "Time for LPTFoveation (full process): %llums", tm.get() );
  Raster::WriteRGB( lptImg, lpt_out );

  Image< PixRGB<byte> > mapImg;
  tm.reset();
  mapImg = lf.getLPT();
  /*  mapImg = LPTFoveator::foveate( img, img.getWidth() / 2,
                                 img.getHeight() / 2, img.getWidth() / 2,
                                 img.getHeight() / 2, true ); */
  LINFO( "Time for LPTFoveation (map image only): %llums", tm.get() );
  Raster::WriteRGB( mapImg, map_out );
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
