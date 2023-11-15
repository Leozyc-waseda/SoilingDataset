/*!@file BeoSub/test-BeoSubPipe.C find pipe     */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubPipe.C $
// $Id: test-BeoSubPipe.C 8542 2007-07-07 21:20:07Z beobot $

#include "Component/ModelManager.H"

#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Media/MediaOpts.H"

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include "GUI/XWinManaged.H"

#include "BeoSub/BeoSubPipe.H"


//END CAMERA STUFF
//canny
#define BOOSTBLURFACTOR 90.0
#define FREE_ARG char*
//#define PI 3.14159
#define FILLBLEED 4
#define INITIAL_TEMPERATURE 30.0
#define FINAL_TEMPERATURE 0.5
#define ALPHA 0.98
#define STEPS_PER_CHANGE 1000

#define BIN_ANGLE 0.588001425

int main(int argc, char* argv[])
{

  MYLOGVERB = LOG_INFO;

  ModelManager manager("ComplexObject Tester");

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "[image {*.ppm}]",
                               0, 1)
      == false) return(1);

  // do post-command-line configs:

  std::string infilename;  //Name of the input image
  bool hasCompIma = false;
  Image<PixRGB<byte> > compIma;
  if(manager.numExtraArgs() > 0)
    {
      infilename = manager.getExtraArgAs<std::string>(0);
      compIma = Raster::ReadRGB(infilename);
      hasCompIma = true;
    }

  int w = ifs->getWidth(),  h = ifs->getHeight();
  std::string dims = convertToString(Dims(w, h));
  LINFO("image size: [%dx%d]", w, h);
  manager.setOptionValString(&OPT_InputFrameDims, dims);

  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  manager.start();
  bool goforever = true;

  rutz::shared_ptr<XWinManaged> inWin;//, resWin;
  inWin.reset(new XWinManaged(Dims(w,h), 0, 0, "Camera Input"));
  //resWin.reset(new XWinManaged(Dims(w,h), w+10, 0, "Pipe Direction Output"));

  // input and output image
  Image< PixRGB<byte> > ima;
  Image< PixRGB<byte> > outputIma;

  rutz::shared_ptr<BeoSubPipe> pipeRecognizer(new BeoSubPipe());
  //(infilename, showMatches, objectname);

  // need an initial image if no comparison image is found
   if(!hasCompIma)
     {
//       ifs->updateNext(); ima = ifs->readRGB();
//       if(!ima.initialized()) goforever = false;
//       else compIma = ima;
     compIma.resize(w,h,NO_INIT);
     }

  std::vector <   Image< PixRGB<byte> > > imageList;

  int cycle = 0;
  while(goforever)
    {
      LINFO("CYCLE %d\n", ++cycle);

//       for(int i = 0; i < 15; i++)
//         {
//           ifs->updateNext();
//           imageList.push_back(ifs->readRGB());
//           if(!(imageList[i]).initialized()) break;
//         }

//       for(uint i = 0; i < imageList.size(); i++)
//         {
//           float angle = pipeRecognizer->pipeOrientation(imageList[i], compIma);
//           LINFO("Pipe Direction: %f", angle);
//           inWin->drawImage(imageList[i],0,0);
//           //resWin->drawImage(outputImg);

//           if(!hasCompIma) compIma = imageList[i];
//         }

//       imageList.clear();

      ifs->updateNext(); ima = ifs->readRGB();
      if(!ima.initialized()) {Raster::waitForKey(); break; }
      inWin->drawImage(ima,0,0);

      float angle = pipeRecognizer->pipeOrientation(ima, compIma);
      LINFO("Pipe Direction: %f", angle);
      //resWin->drawImage(outputImg);

      //      if(!hasCompIma) compIma = ima;


    }

  // get ready to terminate:
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
