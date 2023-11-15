/*!@file Beobot/test-Jet.C Test Jet class */

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
// Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/test-Jet.C $
// $Id: test-Jet.C 12074 2009-11-24 07:51:51Z itti $
//

#include "Beobot/BeobotVisualCortex.H"
#include "GUI/XWindow.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/fancynorm.H"
#include "Raster/Raster.H"

#include <iostream>
#include <cstdio>



#define NBNEIGH 24

// ######################################################################
int main(const int argc, const char **argv)
{
  initRandomNumbers();

  int sml = 4;                       // [4] level of salmap
  int delta_min = 3, delta_max = 4;  // [3,4] spacing (in levs) betw C and S
  int level_min = 2, level_max = 4;  // [2,4] top levels for CS
  int nborient = 4;                  // nb of basis orientations for avg orient
  int jdepth = 3;                    // number of scales in the jets
  int frame, initFrame; char imageNameBase[100]; int outframe = 0;
  bool pause, writeframes, firstFrame = true;

  if (argc >= 2) strcpy(imageNameBase, argv[1]);
  else LFATAL("USAGE: test-Jet <path> [initFrame] [pause] [writeframes]");
  if (imageNameBase[strlen(imageNameBase) - 1] != '/')
    strcat(imageNameBase, "/");
  if (argc >= 3) initFrame = atoi(argv[2]); else initFrame = 0;
  if (argc >= 4) pause = atoi(argv[3]) != 0; else pause = false;
  if (argc >= 5) writeframes = atoi(argv[4]) != 0; else writeframes = false;

  // read the first frame to determine its size
  Image< PixRGB<byte> > col_image =
    Raster::ReadRGB(sformat("%sframe%06d.ppm", imageNameBase, initFrame));

  XWindow xwindow(Dims(col_image.getWidth() * 3, col_image.getHeight() * 2),
                  -1, -1, "Visual Scene Clustering");
  Image< PixRGB<byte> > disp(col_image.getWidth() * 3,
                             col_image.getHeight() * 2, ZEROS);

  Point2D<int> centroid(col_image.getWidth() / (1 << (sml + 1)),
                   col_image.getHeight() / (1 << sml) - 2);
  Point2D<int> prevcentroid(centroid);

  // initialize visual cortex:
  BeobotVisualCortex visualCX;
  visualCX.init(col_image.getWidth(), col_image.getHeight(),
                level_min, level_max, delta_min, delta_max,
                sml, nborient, VCXNORM_DEFAULT, sml, jdepth, NBNEIGH,
                nub::soft_ref<Beowulf>());

  for (frame = initFrame; ; frame++)
    {
      col_image = Raster::ReadRGB(sformat("%sframe%06d.ppm", imageNameBase, frame));
      inplacePaste(disp, col_image, Point2D<int>(0, 0));

      visualCX.newVisualInput(col_image);
      visualCX.process(frame);
      visualCX.initSprings(firstFrame);

      Image< PixRGB<byte> > deformedImage;

      const float dt = 0.1;
      int nbIter; if (firstFrame) nbIter = 5 /*15*/; else nbIter = 5;

      for(int t = 0; t < nbIter; t++)
        {
          // iterate the spring model:
          visualCX.iterateSprings(dt);

          Image< PixRGB<byte> > img;
          visualCX.getPositions(img, 1 << (sml+1));
          inplacePaste(disp, img, Point2D<int>(col_image.getWidth(), 0));

          // compute the clusters:
          visualCX.getClusteredImage(deformedImage, centroid, prevcentroid);
          prevcentroid = centroid;
          inplacePaste(disp, deformedImage, Point2D<int>(0, col_image.getHeight()));

          // display the current state:
          xwindow.drawImage(disp);

          // write frames out:
          if (writeframes) Raster::WriteRGB(disp, sformat("T%06d.ppm", outframe ++));

          if (pause)
            {
              std::cout<<"<<<<< Press [RETURN] to continue >>>>>"<<std::endl;
              getchar();
            }
        }
      firstFrame = false;
    }
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
