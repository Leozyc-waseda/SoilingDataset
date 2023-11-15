/*!@file AppPsycho/psycho-color-calib.C display RGB colors for luminance calibration */

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
// Primary maintainer for this file: David J. Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-color-calib-eval.C $

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/PsychoOpts.H"
#include "Component/ComponentOpts.H"
#include "GUI/SDLdisplay.H"
#include "Neuro/NeuroOpts.H"

#include <vector>


static const ModelOptionDef OPT_RValue =
  { MODOPT_ARG(uint), "RValue", &MOC_DISPLAY, OPTEXP_CORE,
    "intensity value to use in red image [0-255]","rval", '\0', "<uint>", "255" };

static const ModelOptionDef OPT_GValue =
  { MODOPT_ARG(uint), "GValue", &MOC_DISPLAY, OPTEXP_CORE,
    "intensity value to use in green image [0-255]","gval", '\0', "<uint>", "255" };

static const ModelOptionDef OPT_BValue =
  { MODOPT_ARG(uint), "BValue", &MOC_DISPLAY, OPTEXP_CORE,
    "intensity value to use in blue image [0-255]","bval", '\0', "<uint>", "255" };

static const ModelOptionDef OPT_ColDisplayTime =
  { MODOPT_ARG(uint), "DisplayTime", &MOC_DISPLAY, OPTEXP_CORE,
    "time in millesconds to display image","displaytime", '\0', "<uint>", "3000" };

static const ModelOptionDef OPT_RepTimes =
  { MODOPT_ARG(uint), "RepTime", &MOC_DISPLAY, OPTEXP_CORE,
    "times to repeat RGB sequence","RepTime", '\0', "<uint>", "2" };

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages
  
  // Instantiate a ModelManager:
  ModelManager manager("Psycho Color Calibration");

  OModelParam<uint> r(&OPT_RValue, &manager);
  OModelParam<uint> g(&OPT_GValue, &manager);
  OModelParam<uint> b(&OPT_BValue, &manager);
  OModelParam<uint> dtime(&OPT_ColDisplayTime, &manager);
  OModelParam<uint> loops(&OPT_RepTimes, &manager);
  
  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);

  // let's get all our ModelComponent instances started:
  manager.start();
    
  int w = d->getWidth();
  int h = d->getHeight();
  
  std::vector<Image<PixRGB<byte> > > images;
  // r
  images.push_back(Image<PixRGB<byte> >(w,h, NO_INIT));
  images.back().clear(PixRGB<byte>(r.getVal(),0,0));

  //g
  images.push_back(Image<PixRGB<byte> >(w,h, NO_INIT));
  images.back().clear(PixRGB<byte>(0,g.getVal(),0));

  //b
  images.push_back(Image<PixRGB<byte> >(w,h, NO_INIT));
  images.back().clear(PixRGB<byte>(0,0,b.getVal()));

  //rgb
  images.push_back(Image<PixRGB<byte> >(w,h, NO_INIT));
  images.back().clear(PixRGB<byte>(r.getVal(),g.getVal(),b.getVal()));
  
  for (uint cc = 0; cc < loops.getVal(); ++cc)
    for (uint ii = 0; ii < images.size(); ++ii)
      {
        d->displayImage(images[ii],false);
        
        if (dtime.getVal() == 0)
          d->waitForKey();                  
        else
          usleep(dtime.getVal() * 1000);
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
