/*!@file AppNeuro/app-cortical-map.C */

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
// Primary maintainer for this file: David Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/app-cortical-map.C $


#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"

#include "Channels/InputFrame.H"
#include "GUI/XWinManaged.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/ShapeOps.H"
#include "Image/Layout.H"
#include "Image/Normalize.H"
#include "SpaceVariant/FovealTransformModule.H"
#include "SpaceVariant/SpaceVariantDoGModule.H"
#include "SpaceVariant/SpaceVariantEdgeModule.H"
#include "SpaceVariant/SCTransformModule.H"
#include "SpaceVariant/SpaceVariantOpts.H"
#include "Media/FrameSeries.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Util/Pause.H"
#include "Util/csignals.H"
#include "Util/Timer.H"

static const ModelOptionDef OPT_SpaceVariantUseDog =
  { MODOPT_FLAG, "SpaceVariantUseDog", &MOC_SPACEVARIANT, OPTEXP_CORE,
    "display a dog image?","use-dog", '\0', "", "false" };

static const ModelOptionDef OPT_SpaceVariantUseEdge =
  { MODOPT_FLAG, "SpaceVariantUseEdge", &MOC_SPACEVARIANT, OPTEXP_CORE,
    "display an edge image?","use-edge", '\0', "", "false" };

static const ModelOptionDef OPT_SpaceVariantUseLines =
  { MODOPT_FLAG, "SpaceVariantUseLines", &MOC_SPACEVARIANT, OPTEXP_CORE,
    "ignore the input frame series and transform an image of vertical lines", 
    "use-lines", '\0', "", "false" };

static const ModelOptionDef OPT_SpaceVariantSplit =
  { MODOPT_FLAG, "SpaceVariantSplit", &MOC_SPACEVARIANT, OPTEXP_CORE,
    "should we display a fovea and a periphery separately?", 
    "split-fovea-periphery", '\0', "", "false" };

static const ModelOptionDef OPT_UseSC =
  { MODOPT_FLAG, "UseSC", &MOC_SPACEVARIANT, OPTEXP_CORE,
    "should we use he SC transform?", 
    "use-sc", '\0', "", "false" };

int submain(const int argc, const char **argv)
{
  volatile int signum = 0;
  catchsignals(&signum);

  ModelManager manager("Cortical Transform Images");
  OModelParam<bool> useDog(&OPT_SpaceVariantUseDog, &manager);
  OModelParam<bool> useEdge(&OPT_SpaceVariantUseEdge, &manager);
  OModelParam<bool> useLines(&OPT_SpaceVariantUseLines, &manager);
  OModelParam<bool> svSplit(&OPT_SpaceVariantSplit, &manager);
  OModelParam<bool> useSC(&OPT_UseSC, &manager);

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::soft_ref<FovealTransformModule> svm(new FovealTransformModule(manager));
  manager.addSubComponent(svm);

  nub::soft_ref<SpaceVariantDoGModule> svmdog(new SpaceVariantDoGModule(manager));
  manager.addSubComponent(svmdog);

  nub::soft_ref<SpaceVariantEdgeModule> svmedge(new SpaceVariantEdgeModule(manager));
  manager.addSubComponent(svmedge);

  nub::soft_ref<SCTransformModule> sc(new SCTransformModule(manager));
  manager.addSubComponent(sc);

  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);

  manager.start();

  ifs->startStream();

  int c = 0;
  PauseWaiter p;

  SimTime tm = SimTime::ZERO();
  Timer timer(1000);
  while (true)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          return -1;
        }

      if (ofs->becameVoid())
        {
          LINFO("quitting because output stream was closed or became void");
          return 0;
        }

      if (p.checkPause())
        continue;

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      GenericFrame input = ifs->readFrame();
      if (!input.initialized())
        break;

      //display am image of lines   
      if (useLines.getVal())
        {
          Image<PixRGB<byte> > img(input.getDims(), NO_INIT);
          for (int x = 0; x < input.getWidth(); ++x)
            for (int y = 0; y < input.getHeight(); ++y)
              if (x % 2 == 0)
                img.setVal(x,y,PixRGB<byte>(0,0,255));
              else
                img.setVal(x,y,PixRGB<byte>(255,0,0));
          input = GenericFrame(img);
        }
      
      Image<PixRGB<byte> > ctimage, ctimageinv;
      if (useDog.getVal())
        {
          //get rgb image and transform
          const Image<float> rgbin = input.asGray();
          timer.reset();
          ctimage = normalizeFloat(svmdog->transformDoG(rgbin), FLOAT_NORM_0_255);
          LINFO("Transform time: %.6f", timer.getSecs());
          timer.reset();
          ctimageinv = svmdog->inverseTransform(ctimage);
          LINFO("Inverse Transform time: %.6f", timer.getSecs()); 
        }
      else if (useEdge.getVal())
        {
          //get rgb image and transform
          const Image<float> rgbin = input.asGray();
          timer.reset();
          ctimage = normalizeFloat(svmedge->transformEdge(rgbin), FLOAT_NORM_0_255);

          LINFO("Transform time: %.6f", timer.getSecs());
          timer.reset();
          ctimageinv = svmedge->inverseTransform(ctimage);
          LINFO("Inverse Transform time: %.6f", timer.getSecs());
        }
      else if (useSC.getVal())
      {
          //get rgb image and transform
          const Image<PixRGB<byte> > rgbin = input.asRgb();
          timer.reset();
          ctimage = sc->transform(rgbin);

          LINFO("Transform time: %.6f", timer.getSecs());
          timer.reset();
          ctimageinv = sc->inverseTransform(ctimage);
          LINFO("Inverse Transform time: %.6f", timer.getSecs());
      }
      else
        {
          //get rgb image and transform
          const Image<PixRGB<byte> > rgbin = input.asRgb();
          timer.reset();
          ctimage = svm->transform(rgbin);

          LINFO("Transform time: %.6f", timer.getSecs());
          timer.reset();
          ctimageinv = svm->inverseTransform(ctimage);
          LINFO("Inverse Transform time: %.6f", timer.getSecs());
        }
      

      
      const FrameState os = ofs->updateNext();

      if (svSplit.getVal() && !useSC.getVal())
        {
          Image<PixRGB<byte> > f, p;
          svm->getFoveaPeriphery(ctimage, f, p);
          ofs->writeRGB(f,"Cortical Image Fovea"); 
          ofs->writeRGB(p,"Cortical Image Periphery"); 
        }
      else
        ofs->writeRGB(ctimage,"Cortical Image");
      
      ofs->writeRGB(ctimageinv,"Cortical Image Inverse");
      ofs->writeFrame(input,"input");
      
      if (os == FRAME_FINAL)
        break;
      
      LDEBUG("frame %d", c++);
      
      if (ifs->shouldWait() || ofs->shouldWait())
        Raster::waitForKey();
      
      tm += SimTime::HERTZ(30);
    }

  return 0;
}

int main(const int argc, const char **argv)
{
  try
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }

  return 1;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
