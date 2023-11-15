/*!@file AppPsycho/app-randomize-luminance.C Generates stimuli with randomized
  luminance drawn from a uniform distribution with mean based on
  standard flicker photometry results, and variance 5*/

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
// Primary maintainer for this file: Vidhya Navalpakkam <navalpak@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/app-randomize-luminance.C $
// $Id: app-randomize-luminance.C 12074 2009-11-24 07:51:51Z itti $
//

#include "Image/Image.H"
#include "Component/ModelManager.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include <cstdio>

// ######################################################################

int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Generate randomized luminance stimuli");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<target.ppm> "
                               " <luminRef> <lumin2> <lumin3>"
                               " <satRef> <sat2> <sat3>"
                               " <perceptIncrement>", 1, -1)==false)
    return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // load up the frame and show a fixation cross on a blank screen:
  LINFO("Loading '%s'...", manager.getExtraArg(0).c_str());
  Image< PixRGB<byte> > stim =
    Raster::ReadRGB(manager.getExtraArg(0));

  // read the luminance and saturation value of S1, S2 and S3
  int lumin[3]; float sat[3];
  lumin[0] = manager.getExtraArgAs<int>(1);
  lumin[1] = manager.getExtraArgAs<int>(2);
  lumin[2] = manager.getExtraArgAs<int>(3);
  sat[0] = manager.getExtraArgAs<float>(4);
  sat[1] = manager.getExtraArgAs<float>(5);
  sat[2] = manager.getExtraArgAs<float>(6);
  const char* name = manager.getExtraArg(7).c_str();
  float perceptIncrement = manager.getExtraArgAs<float>(8);
  int perceptLum = 0, refLum = 0;
  // generate stimuli with randomized luminance
  for (int i = 0; i < 3; i ++){
    for (int j = 0; j < 3; j++) {
      // generate stimuli with the following lumin based on weber's law
      if (j == 0){
        perceptLum = (int) (lumin[i]/(1+perceptIncrement));
        refLum = (int) (lumin[0]/(1+perceptIncrement));
      }
      else if (j == 1){
        perceptLum = lumin[i];
        refLum = lumin[0];
      }
      else if (j == 2){
        perceptLum = (int) (lumin[i] * (1+perceptIncrement));
        refLum = (int) (lumin[0] * (1+perceptIncrement));
      }
      // generate the r, g, b values for drawing the stimuli
      int red = (int) (perceptLum / (1.0f - 0.7875*sat[i]));
      int green = (int) ((perceptLum - 0.2125*red)/0.7875);
      PixRGB<byte> zero(0, 0, 0);
      // since the stimuli is red saturated, blue = green
      PixRGB<byte> rgb(red, green, green);
      int w = stim.getWidth(), h = stim.getHeight();
      Image< PixRGB<byte> > target(w, h, ZEROS);
      // regenerate the image
      for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y ++) {
          if (stim.getVal(x,y) == zero);  // do nothing
          else {
            stim.setVal(x, y, rgb);
            // target undergoes 180 deg. rotation and (w,h)px translation
            target.setVal(w-x, h-y, rgb);
          }
        }
      // print this image to file
      char fileName[30];
      sprintf (fileName, "%s_%.2f_%d.ppm", name, sat[i], refLum);
      LINFO ("%s: [r,g,b] =  %d, %d, %d", fileName, red, green, green);
      Raster::WriteRGB (stim, fileName);
      sprintf (fileName, "t_sat_%.2f_%d.ppm", sat[i], refLum);
      LINFO ("%s: [r,g,b] =  %d, %d, %d", fileName, red, green, green);
      Raster::WriteRGB (target, fileName);
     }
  }

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
