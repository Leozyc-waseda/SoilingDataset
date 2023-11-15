/*!@file NeovisionII/objRec-ServerPCA.C */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/objRec-serverPCA.C $
// $Id: objRec-serverPCA.C 13901 2010-09-09 15:12:26Z lior $
//

#ifndef OBJREC_SERVERPCA_C_DEFINED
#define OBJREC_SERVERPCA_C_DEFINED

#include "Image/OpenCVUtil.H"
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/ShapeOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/ColorOps.H"
#include "Image/Transforms.H"
#include "Image/MathOps.H"
#include "Learn/Bayes.H"
#include "GUI/DebugWin.H"
#include "SIFT/ScaleSpace.H"
#include "SIFT/VisualObject.H"
#include "SIFT/Keypoint.H"
#include "SIFT/VisualObjectDB.H"
#include "NeovisionII/nv2_common.h"
#include "NeovisionII/nv2_label_server.h"
#include "rutz/fstring.h"
#include "rutz/time.h"
#include "rutz/timeformat.h"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "GUI/XWinManaged.H"
#include "GUI/ImageDisplayStream.H"

#include <iostream> // for std::cin

const bool USECOLOR = false;

bool terminate = false;

void terminateProc(int s)
{
  terminate = true;
}

std::string getBestLabel(const std::deque<std::string>& labels,
                         const size_t mincount)
{
  if (labels.size() == 0)
    return std::string();

  std::map<std::string, size_t> counts;

  size_t bestcount = 0;
  size_t bestpos = 0;

  for (size_t i = 0; i < labels.size(); ++i)
    {
      const size_t c = ++(counts[labels[i]]);

      if (c >= bestcount)
        {
          bestcount = c;
          bestpos = i;
        }
    }

  if (bestcount >= mincount)
    return labels[bestpos];

  return std::string();
}

namespace
{
  void fillRegion(Image<PixRGB<byte> >& img, PixRGB<byte> col,
                  const int x0, const int x1,
                  const int y0, const int y1)
  {
    for (int x = x0; x < x1; ++x)
      for (int y = y0; y < y1; ++y)
        img.setVal(x, y, col);
  }

  Image<PixRGB<byte> > makeColorbars(const int w, const int h)
  {
    Image<PixRGB<byte> > result = Image<PixRGB<byte> >(w, h, ZEROS);

    const PixRGB<byte> cols[] =
      {
        PixRGB<byte>(255, 255, 255), // white
        PixRGB<byte>(255, 255, 0),   // yellow
        PixRGB<byte>(0,   255, 255), // cyan
        PixRGB<byte>(0,   255, 0),   // green
        PixRGB<byte>(255, 0,   255), // magenta
        PixRGB<byte>(255, 0,   0),   // red
        PixRGB<byte>(0,   0,   255)  // blue
      };

    int x1 = 0;
    for (int i = 0; i < 7; ++i)
      {
        const int x0 = x1+1;
        x1 = int(double(w)*(i+1)/7.0 + 0.5);
        fillRegion(result, cols[i],
                   x0, x1,
                   0, int(h*2.0/3.0));
      }

    x1 = 0;
    for (int i = 0; i < 16; ++i)
      {
        const int x0 = x1;
        x1 = int(double(w)*(i+1)/16.0 + 0.5);
        const int gray = int(255.0*i/15.0 + 0.5);
        fillRegion(result, PixRGB<byte>(gray, gray, gray),
                   x0, x1,
                   int(h*2.0/3.0)+1, int(h*5.0/6.0));
      }

    fillRegion(result, PixRGB<byte>(255, 0, 0),
               0, w,
               int(h*5.0/6.0)+1, h);

    writeText(result, Point2D<int>(1, int(h*5.0/6.0)+2),
              "iLab Neuromorphic Vision",
              PixRGB<byte>(0, 0, 0), PixRGB<byte>(255, 0, 0),
              SimpleFont::FIXED(10));

    return result;
  }

  Image<PixRGB<byte> > addLabels(const Image<PixRGB<byte> >& templ,
                                 const int fnum)
  {
    Image<PixRGB<byte> > result = templ;

    std::string fnumstr = sformat("%06d", fnum);
    writeText(result, Point2D<int>(1, 1),
              fnumstr.c_str(),
              PixRGB<byte>(0, 0, 0), PixRGB<byte>(255, 255, 255),
              SimpleFont::FIXED(10));

    rutz::time t = rutz::time::wall_clock_now();

    writeText(result, Point2D<int>(1, result.getHeight() - 14),
              rutz::format_time(t).c_str(),
              PixRGB<byte>(32, 32, 32), PixRGB<byte>(255, 0, 0),
              SimpleFont::FIXED(6));

    return result;
  }
}

void trainPCA(ImageSet<byte> images)
{
  int imagesCollected = 10;
  IplImage* input[imagesCollected];

  //IplImage* input = img2ipl(images[0]);
  CvMat* pcaInputs = cvCreateMat(imagesCollected, (input[0]->width * input[0]->height), CV_8UC1);
  CvMat* average = cvCreateMat(1, (input[0]->width * input[0]->height), CV_32FC1);
  CvMat* eigenValues = cvCreateMat(1, std::min(pcaInputs->rows, pcaInputs->cols), CV_32FC1);
  CvMat* eigens = cvCreateMat(imagesCollected, (input[0]->width * input[0]->height), CV_32FC1);
  CvMat* coefficients = cvCreateMat(imagesCollected, eigens->rows, CV_32FC1);

  // construct required structures for later recognition

  CvMat* recogniseCoeffs = cvCreateMat(1, eigens->rows, CV_32FC1);
  CvMat* recognise = cvCreateMat(1, input[0]->width * input[0]->height, CV_8UC1);

  for (int i = 0; i < imagesCollected; i++){
    for (int j = 0; j < (input[0]->width * input[0]->height); j++){
      CV_MAT_ELEM(*pcaInputs, uchar, i, j) = (input[i])->imageData[(j)];
    }
  }

  // compute eigen image representation

  cvCalcPCA(pcaInputs, average, eigenValues, eigens, CV_PCA_DATA_AS_ROW);

  // compute eigen. co-efficients for all sample images and store

  cvProjectPCA(pcaInputs, average, eigens, coefficients);

  for (int i = 0; i < imagesCollected; i++){cvReleaseImage( &(input[i]));}

  // release matrix objects

  cvReleaseMat( &pcaInputs);
  cvReleaseMat( &average );
  cvReleaseMat( &eigenValues );
  cvReleaseMat( &eigens );
  cvReleaseMat( &coefficients );
  cvReleaseMat( &recogniseCoeffs );
  cvReleaseMat( &recognise );

}

std::string recogPCA(Image<byte> img)
{

  // project image to eigen space


//  for (int j = 0; j < (input[0]->width * input[0]->height); j++){
//    CV_MAT_ELEM(*recognise, uchar, 0, j) = (grayImg)->imageData[(j)];
//  }
//
//  cvProjectPCA(recognise, average, eigens, recogniseCoeffs);
//
//  // check which set of stored sample co-efficients it is
//  // closest too and then display the corresponding image
//
//  double closestCoeffDistance = HUGE;
//  int closestImage = 0;
//
//  for (int i = 0; i < imagesCollected; i++)
//  {
//    double diff = 0;
//    for(int j = 0; j < recogniseCoeffs->cols; j++)
//    {
//      diff += fabs(CV_MAT_ELEM(*coefficients, float, i, j)
//          - CV_MAT_ELEM(*recogniseCoeffs, float, 0, j));
//    }
//    if (diff < closestCoeffDistance){
//      closestCoeffDistance = diff;
//      closestImage = i;
//
//    }
//  }

  return std::string("nomatch");
}


Point2D<int> getMouseClick(nub::ref<OutputFrameSeries> &ofs, const char* wname)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow(wname)
    : rutz::shared_ptr<XWinManaged>();

  if (uiwin.is_valid())
    return uiwin->getLastMouseClick();
  else
    return Point2D<int>(-1,-1);
}

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager mgr("Test ObjRec");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(mgr));
  mgr.addSubComponent(ofs);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(mgr));
  mgr.addSubComponent(ifs);




  if (mgr.parseCommandLine(argc, argv, "<vdb file> <server ip>", 2, 2) == false)
    return 1;

  mgr.start();

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminateProc); signal(SIGINT, terminateProc);
  signal(SIGQUIT, terminateProc); signal(SIGTERM, terminateProc);
  signal(SIGALRM, terminateProc);

  //get command line options
  const std::string vdbFile = mgr.getExtraArg(0);
  const std::string server_ip = mgr.getExtraArg(1);
  bool train = false;

  struct nv2_label_server* labelServer =
    nv2_label_server_create(9930,
                            server_ip.c_str(),
                            9931);

  nv2_label_server_set_verbosity(labelServer,1); //allow warnings


  const size_t max_label_history = 1;
  std::deque<std::string> recent_labels;

  Image<PixRGB<byte> > colorbars = makeColorbars(256, 256);

  bool getImgFromFile = true;
  while (!terminate)
  {

    Image<PixRGB<byte> > inputImg;
    struct nv2_image_patch p;

    if (getImgFromFile)
    {

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE) return 0;
      GenericFrame input = ifs->readFrame();
      inputImg = input.asRgb();
    } else {

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
        memcpy(img.getArrayPtr(), p.data, p.width*p.height*3);

        inputImg = rescale(img, 256, 256);
      }
    }

    float score = 0;
    std::string objName = "nomatch";

    if (inputImg.initialized())
    {
      ofs->writeRGB(inputImg, "object", FrameInfo("object", SRC_POS));
      getchar();
      ofs->updateNext();

      Point2D<int> clickLoc = getMouseClick(ofs, "object");
      if (clickLoc.isValid())
        train = !train;


      if (objName == "nomatch")
      {
        recent_labels.resize(0);

        if (train)
        {
          printf("Enter a label for this object:\n");
          std::getline(std::cin, objName);
          printf("You typed '%s'\n", objName.c_str());

          if (objName == "exit")
            break;
          else if (objName != "")
          {
            //Train object with objName
          }
        }
      }
    }

    if (objName != "nomatch" && !getImgFromFile)
    {
      recent_labels.push_back(objName);
      while (recent_labels.size() > max_label_history)
        recent_labels.pop_front();

      const std::string bestObjName =
        getBestLabel(recent_labels, 1);

      if (bestObjName.size() > 0)
      {
        struct nv2_patch_label l;
        l.protocol_version = NV2_LABEL_PROTOCOL_VERSION;
        l.patch_id = p.id;
        l.confidence = (int)(score*100.0F);
        snprintf(l.source, sizeof(l.source), "%s",
            "ObjRec");
        snprintf(l.name, sizeof(l.name), "%s",
            objName.c_str());
        snprintf(l.extra_info, sizeof(l.extra_info),
            "%ux%u #%u",
            (unsigned int) p.width,
            (unsigned int) p.height,
            (unsigned int) p.id);

        nv2_label_server_send_label(labelServer, &l);

        LINFO("sent label '%s (%s)'\n", l.name, l.extra_info);
      }
      nv2_image_patch_destroy(&p);
    }
  }

  if (terminate)
    LINFO("Ending application because a signal was caught");

  nv2_label_server_destroy(labelServer);
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
