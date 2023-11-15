/*!@file Beosub/BeeBrain/test-stereoVision.C for stereo vision          */
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/test-stereoVision.C $
// $Id: test-stereoVision.C 8623 2007-07-25 17:57:51Z rjpeters $

#define NAVG 20

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Video/VideoFormat.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/FrameGrabberFactory.H"
#include "GUI/XWinManaged.H"

#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/Pixels.H"

#include "Util/Timer.H"

int main( int argc, const char* argv[] )
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("test stereo vision");

  nub::soft_ref<FrameIstream> gbF(makeV4L2grabber(manager));
  nub::soft_ref<FrameIstream> gbB(makeV4L2grabber(manager));

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv," ",0, 0) == false) return(1);

  gbF->setModelParamVal("FrameGrabberDevice",std::string("/dev/video0"));
  gbF->setModelParamVal("FrameGrabberChannel",0);
  //gbF->setModelParamVal("InputFrameSource", "V4L2");
  gbF->setModelParamVal("FrameGrabberMode", VIDFMT_YUYV);
  gbF->setModelParamVal("FrameGrabberByteSwap", false);
  //gbF->setModelParamVal("FrameGrabberFPS", 30);
  manager.addSubComponent(gbF);

  gbB->setModelParamVal("FrameGrabberDevice",std::string("/dev/video1"));
  gbB->setModelParamVal("FrameGrabberChannel",0);
  //gbB->setModelParamVal("InputFrameSource", "V4L2");
  gbB->setModelParamVal("FrameGrabberMode", VIDFMT_YUYV);
  gbB->setModelParamVal("FrameGrabberByteSwap", false);
  //gbB->setModelParamVal("FrameGrabberFPS", 30);
  manager.addSubComponent(gbB);

  manager.start();

  uint wF = gbF->getWidth(); uint hF = gbF->getHeight();
  uint wB = gbB->getWidth(); uint hB = gbB->getHeight();

  rutz::shared_ptr<XWinManaged> fwin(new XWinManaged(Dims(wF,hF), 0, 20, "front"));
  rutz::shared_ptr<XWinManaged> bwin(new XWinManaged(Dims(wB,hB), wF,20, "back"));

  //start streaming
  gbF->startStream();
  gbB->startStream();

  Timer tim(1000000); uint64 t[NAVG]; float frate = 0.0f; tim.reset();
  bool goforever =true;
  uint fNum =0;
  while(goforever)
    {
      //grab the images
      Image< PixRGB<byte> > fImg = gbF->readRGB();
      Image< PixRGB<byte> > bImg = gbB->readRGB();

      if((fNum % 15) ==0) fwin->drawImage(fImg,0,0);
      if((fNum % 15) ==0) bwin->drawImage(bImg,0,0);

      // compute and show framerate over the last NAVG frames:
      t[fNum % NAVG] = tim.get(); tim.reset();
      if (fNum % NAVG == 0 && fNum > 0)
        {
          uint64 avg = 0ULL;
          for(int i = 0; i < NAVG; i ++) avg += t[i];
          frate = 1000000.0F / float(avg) * float(NAVG);
          LINFO("[%6d] Frame rate: %6.3f fps -> %8.3f ms/frame",
                fNum,frate, 1000.0/frate);
        }

      //std::string saveFNameF = sformat("data/cross_F_%07d.ppm", fNum);
      //std::string saveFNameB = sformat("data/cross_B_%07d.ppm", fNum);
      //LINFO("saving: %s & %s",saveFNameF.c_str(),saveFNameB.c_str());
      //Raster::WriteRGB(fImg,saveFNameF);
      //Raster::WriteRGB(bImg,saveFNameB);
      fNum++;
    }

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */


