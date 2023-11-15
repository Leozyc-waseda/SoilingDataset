/*!@file BeoSub/test-BeoSubTaskDecoder.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubTaskDecoder.C $
// $Id: test-BeoSubTaskDecoder.C 8521 2007-06-28 17:45:49Z rjpeters $
//

#ifndef TESTBEOSUBTASKDECODER_H_DEFINED
#define TESTBEOSUBTASKDECODER_H_DEFINED


#include "Component/ModelManager.H"

#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Media/MediaOpts.H"

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include "GUI/XWinManaged.H"

#include "BeoSub/BeoSubTaskDecoder.H"

#include "Image/ImageSet.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <string>

#define NAVG 20

int main(int argc, char **argv)
{

  ModelManager manager("Frame Grabber Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::soft_ref<BeoSubTaskDecoder> test(new BeoSubTaskDecoder(manager));
  manager.addSubComponent(test);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  int w = ifs->getWidth(),  h = ifs->getHeight();
  std::string dims = convertToString(Dims(w, h));
  LINFO("image size: [%dx%d]", w, h);
  manager.setOptionValString(&OPT_InputFrameDims, dims);
  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  // let's get all our ModelComponent instances started:
  manager.start();

  // get ready for main loop:
  rutz::shared_ptr<XWinManaged>  win(new XWinManaged(Dims(w,h), 200, 200, "input window"));
  Timer tim; uint64 t[NAVG]; int frame = 0;

  // get the frame grabber to start streaming:


  //Parse the command line options
  //  char *colorArg = NULL;    //Color for tracking

  //if(argc < 2){
  //  fprintf(stderr,"\n<USAGE> %s color \n",argv[0]);
  //  fprintf(stderr,"      color:       Color to track\n");
  //  fprintf(stderr,"                  Candidates: Green, Red\n");
  //  exit(1);
  //}
  //colorArg = argv[1];
  //printf("READ: 1: %s\n", colorArg);





 //Load in config file for camera FIX: put in a check whether config file exists!
  //mgr.loadConfig("camconfig.pmap");




   //mgr.start();


  float avg2 = 0.0;

  Image< PixRGB<byte> > img;

  bool goforever = true;
  while(goforever){

    ImageSet< PixRGB<byte> > input;
    for(int i = 0; i < 100; i++){
      tim.reset();

      ifs->updateNext(); img = ifs->readRGB();
      if(!img.initialized()) break;
      win->drawImage(img,0,0);
      input.push_back(img);

      char buf[32];
      sprintf(buf, "image%d.ppm", i);
      Raster::WriteRGB(img, buf);

      uint64 t0 = tim.get();  // to measure display time
      t[frame % NAVG] = tim.get();
      t0 = t[frame % NAVG] - t0;
      // compute and show framerate over the last NAVG frames:
      if (frame % NAVG == 0 && frame > 0)
        {
          uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
          avg2 = 1000.0F / float(avg) * float(NAVG);
        }
      frame ++;
     }

    LINFO("fps: %f", avg2);
    test->setupDecoder("Red", true);
    test->runDecoder(input, avg2);
    float hertz = test->calculateHz();

    printf("\n\nFinal hertz calculated is: %f\n\n", hertz);
    Raster::waitForKey();
  }

  manager.stop();
  return 0;
}

#endif
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
