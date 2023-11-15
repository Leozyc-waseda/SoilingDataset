/*!@file NeovisionII/objRec-ServerSift.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/objRec-serverSift.C $
// $Id: objRec-serverSift.C 13901 2010-09-09 15:12:26Z lior $
//

#ifndef OBJREC_SERVERSIFT_C_DEFINED
#define OBJREC_SERVERSIFT_C_DEFINED

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

#include <iostream> // for std::cin

const bool USECOLOR = false;

bool terminate = false;

void terminateProc(int s)
{
  terminate = true;
}

std::string matchObject(Image<PixRGB<byte> > &ima, VisualObjectDB& vdb, float &score)
{
  //find object in the database
  std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
  rutz::shared_ptr<VisualObject>
    vo(new VisualObject("PIC", "PIC", ima,
                        Point2D<int>(-1,-1),
                        std::vector<float>(),
                        std::vector< rutz::shared_ptr<Keypoint> >(),
                        USECOLOR));

  const uint nmatches = vdb.getObjectMatches(vo, matches, VOMA_SIMPLE,
                                             100U, //max objs to return
                                             0.5F, //keypoint distance score default 0.5F
                                             0.5F, //affine distance score default 0.5F
                                             1.0F, //minscore  default 1.0F
                                             3U, //min # of keypoint match
                                             100U, //keypoint selection thershold
                                             false //sort by preattentive
                                             );

  score = 0;
  float avgScore = 0, affineAvgDist = 0;
  int nkeyp = 0;
  int objId = -1;
  if (nmatches > 0)
    {
      rutz::shared_ptr<VisualObject> obj; //so we will have a ref to the last matches obj
      rutz::shared_ptr<VisualObjectMatch> vom;
      //for(unsigned int i=0; i< nmatches; i++){
      for (unsigned int i = 0; i < 1; ++i)
        {
          vom = matches[i];
          obj = vom->getVoTest();
          score = vom->getScore();
          nkeyp = vom->size();
          avgScore = vom->getKeypointAvgDist();
          affineAvgDist = vom->getAffineAvgDist();

          objId = atoi(obj->getName().c_str()+3);

          return obj->getName();
          LINFO("### Object match with '%s' score=%f ID:%i",
                obj->getName().c_str(), vom->getScore(), objId);

          //calculate the actual distance (location of keypoints) between
          //keypoints. If the same patch was found, then the distance should
          //be close to 0
          double dist = 0;
          for (int keyp=0; keyp<nkeyp; keyp++)
            {
              const KeypointMatch kpm = vom->getKeypointMatch(keyp);

              float refX = kpm.refkp->getX();
              float refY = kpm.refkp->getY();

              float tstX = kpm.tstkp->getX();
              float tstY = kpm.tstkp->getY();
              dist += (refX-tstX) * (refX-tstX);
              dist += (refY-tstY) * (refY-tstY);
            }

          //   printf("%i:%s %i %f %i %f %f %f\n", objNum, obj->getName().c_str(),
          //       nmatches, score, nkeyp, avgScore, affineAvgDist, sqrt(dist));

          //analizeImage();
        }

    }

  return std::string("nomatch");
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

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager mgr("Test ObjRec");

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

  LINFO("Loading db from %s\n", vdbFile.c_str());
  VisualObjectDB vdb;
  vdb.loadFrom(vdbFile);

  XWinManaged xwin(Dims(256,256),
                   -1, -1, "ILab Robot Head Demo");


  struct nv2_label_server* labelServer =
    nv2_label_server_create(9930,
                            server_ip.c_str(),
                            9931);

  nv2_label_server_set_verbosity(labelServer,1); //allow warnings


  const size_t max_label_history = 1;
  std::deque<std::string> recent_labels;

  Image<PixRGB<byte> > colorbars = makeColorbars(256, 256);

  while (!terminate)
    {
      Point2D<int> clickLoc = xwin.getLastMouseClick();
      if (clickLoc.isValid())
        train = !train;

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
              xwin.drawImage(addLabels(colorbars, p.id));
              continue;
            }

          Image<PixRGB<byte> > img(p.width, p.height, NO_INIT);
          memcpy(img.getArrayPtr(), p.data, p.width*p.height*3);

          Image<PixRGB<byte> > inputImg = rescale(img, 256, 256);

          xwin.drawImage(inputImg);
          float score = 0;
          std::string objName = matchObject(inputImg, vdb, score);

          if (objName == "nomatch")
            {
              recent_labels.resize(0);

              if (train)
                {
                  LINFO("Enter a label for this object:\n");
                  std::getline(std::cin, objName);
                  LINFO("You typed '%s'\n", objName.c_str());

                  if (objName == "exit")
                    break;
                  else if (objName != "")
                    {
                      rutz::shared_ptr<VisualObject>
                        vo(new VisualObject(objName.c_str(), "NULL", inputImg,
                                            Point2D<int>(-1,-1),
                                            std::vector<float>(),
                                            std::vector< rutz::shared_ptr<Keypoint> >(),
                                            USECOLOR));
                      vdb.addObject(vo);
                      vdb.saveTo(vdbFile);
                    }
                }
            }
          else
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
