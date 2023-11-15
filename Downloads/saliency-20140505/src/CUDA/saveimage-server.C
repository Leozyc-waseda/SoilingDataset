/*!@file HMAX/test-hmax5.C Test Hmax class and compare to original code */

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
// Primary maintainer for this file: Dan Parks <danielfp@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/saveimage-server.C $
// $Id: saveimage-server.C 14154 2010-10-21 05:07:25Z dparks $
//

#include "Component/ModelManager.H"
#include "Learn/Bayes.H"
#include "GUI/DebugWin.H"
#include "NeovisionII/nv2_common.h"
#include "NeovisionII/nv2_label_server.h"

#include "GUI/XWindow.H"
#include "CUDA/CudaHmaxFL.H"
#include "CUDA/CudaHmax.H"
#include "CUDA/CudaImage.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/ColorOps.H"
#include "Image/ShapeOps.H"
#include "Image/CutPaste.H"
#include "Image/FilterOps.H"
#include "Image/Rectangle.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/MatrixOps.H"
#include "Image/Transforms.H"
#include "Image/Convolutions.H"
#include "Learn/SVMClassifier.H"
#include "Media/FrameSeries.H"
#include "nub/ref.h"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <signal.h>

#include "rutz/fstring.h"
#include "rutz/time.h"
#include "rutz/timeformat.h"


#include <fstream>
#include <map>
#include <vector>
#include <utility>
#include <iostream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <cstdlib>


// number of orientations to use in HmaxFL
#define NORI 4
#define NUM_PATCHES_PER_SIZE 250


const bool USECOLOR = false;

bool terminate = false;

void terminateProc(int s)
{
  terminate = true;
}


int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;

  ModelManager *mgr = new ModelManager("Hmax with Feature Learning Server");


  mgr->exportOptions(MC_RECURSE);


  if (mgr->parseCommandLine(
                            (const int)argc, (const char**)argv, "<sampleimagesdir> <localport> <server_ip> <serverport> <prefix>", 4, 5) == false)
    return 1;

  std::string serverIP,serverPortStr,localPortStr;
  std::string sampleImagesDir,prefixStr;

  // Now we run
  mgr->start();

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminateProc); signal(SIGINT, terminateProc);
  signal(SIGQUIT, terminateProc); signal(SIGTERM, terminateProc);
  signal(SIGALRM, terminateProc);


  sampleImagesDir = mgr->getExtraArg(0);
  localPortStr = mgr->getExtraArg(1);
  serverIP = mgr->getExtraArg(2);
  serverPortStr = mgr->getExtraArg(3);
  prefixStr = std::string("Sample");
  if(mgr->numExtraArgs() == 5)
    prefixStr = mgr->getExtraArg(4);

  XWinManaged xwin(Dims(256,256),
                   -1, -1, "ILab Robot Head Demo");

  int serverPort = strtol(serverPortStr.c_str(),NULL,0);
  int localPort = strtol(localPortStr.c_str(),NULL,0);
  struct nv2_label_server* labelServer =
    nv2_label_server_create(localPort,
                            serverIP.c_str(),
                            serverPort);

  nv2_label_server_set_verbosity(labelServer,1); //allow warnings
  int curPatch = 0;

  //const size_t max_label_history = 1;
  //std::deque<std::string> recent_labels;

  //Image<PixRGB<byte> > colorbars = makeColorbars(256, 256);


  while(!terminate)
    {
      //Point2D<int> clickLoc = xwin.getLastMouseClick();

      struct nv2_image_patch p;
      const enum nv2_image_patch_result res =
        nv2_label_server_get_current_patch(labelServer, &p);

      std::string objName;
      if (res == NV2_IMAGE_PATCH_END)
        {
          LINFO("ok, quitting");
          break;
        }
      else if (res == NV2_IMAGE_PATCH_NONE)
        {
          usleep(10000);
          continue;
        }
      else if (res == NV2_IMAGE_PATCH_VALID)
        {
          if (p.type != NV2_PIXEL_TYPE_RGB24)
            {
              LINFO("got a non-rgb24 patch; ignoring %i", p.type);
              continue;
            }

          if (p.width * p.height == 1)
            {
              //xwin.drawImage(addLabels(colorbars, p.id));
              continue;
            }

          Image<PixRGB<byte> > img(p.width, p.height, NO_INIT);
          // Get the test image from the socket
          memcpy(img.getArrayPtr(), p.data, p.width*p.height*3);

          Image<PixRGB<byte> > inputImg = rescale(img, 256, 256);

          //xwin.drawImage(inputImg);

          //Image<float> inputf = luminance(inputImg);
          char fname[200];
          sprintf(fname,"%s/%s-%d.png",sampleImagesDir.c_str(),prefixStr.c_str(),curPatch);
          Raster::WriteRGB(inputImg,fname);
          curPatch++;
        }
    }
  if (terminate)
    LINFO("Ending application because a signal was caught");

  //nv2_label_server_destroy(labelServer);
  LINFO("Got Here");

  return 0;
}




// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
