/*!@file RCBot/trackSIFT.C track featuers */

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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/trackSIFT.C $
// $Id: trackSIFT.C 13993 2010-09-20 04:54:23Z itti $
//

#include "Component/ModelManager.H"
#include "Component/OptionManager.H"
#include "Controllers/PID.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/sc8000.H"
#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Neuro/SaliencyMap.H"
#include "Neuro/VisualCortex.H"
#include "Neuro/WTAwinner.H"
#include "Neuro/WinnerTakeAll.H"
#include "RCBot/Motion/MotionEnergy.H"
#include "Raster/Raster.H"
#include "SIFT/Keypoint.H"
#include "SIFT/ScaleSpace.H"
#include "SIFT/VisualObject.H"
#include "SIFT/VisualObjectDB.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include <math.h>


#define UP_KEY 98
#define DOWN_KEY 104
#define LEFT_KEY 100
#define RIGHT_KEY 102

XWinManaged window(Dims(256, 256), -1, -1, "Test Output 1");
XWinManaged window1(Dims(256, 256), -1, -1, "S Map");

// ######################################################################
int main(int argc, char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Camera capture");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<VisualCortex>
    itsVC(new VisualCortex(manager, "Visual Cortex", "VC"));

  /* FIXME
  VisualCortexWeights wts = VisualCortexWeights::zeros();
  wts.chanIw = 0.1;
  wts.chanCw = 0.1;
  wts.chanOw = 1.0;
  itsVC->addDefaultChannels(wts);
  */
  manager.addSubComponent(itsVC);

  nub::soft_ref<SaliencyMap> itsSMAP(new SaliencyMapTrivial(manager));
  manager.addSubComponent(itsSMAP);

  nub::soft_ref<WinnerTakeAll> itsWTA(new WinnerTakeAllFast(manager));
  manager.addSubComponent(itsWTA);


  // Parse command-line:
  if (manager.parseCommandLine(argc, (const char **)argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.get() == NULL)
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful -- ABORT");
  // int w = gb->getWidth(), h = gb->getHeight();

  //manager.setModelParamVal("InputFrameDims", Dims(w, h), MC_RECURSE | MC_IGNORE_MISSING);

  // let's get all our ModelComponent instances started:
  manager.start();


  // get the frame grabber to start streaming:
  gb->startStream();


  // ########## MAIN LOOP: grab, process, display:
  int key = 0;

  //sc8000->move(3, -0.3); //move forward slowly

  double time = 0;
  itsVC->reset(MC_RECURSE);
  std::vector<double> feature_vector;

  std::vector<double> mean;

  Image<PixRGB<byte> > track_img;
  Image<PixRGB<byte> > key_points;

  VisualObjectDB vdb;
  //vdb.loadFrom("test.vdb");


  bool find_obj = false;

  while(key != 24){
    // receive conspicuity maps:
    // grab an image:

    Image< PixRGB<byte> > ima = gb->readRGB();

    Point2D<int> location = window.getLastMouseClick();
    if (location.i > -1 && location.j > -1){
      Dims WindowDims = window.getDims();

      float newi = (float)location.i * (float)ima.getWidth()/(float)WindowDims.w();
      float newj = (float)location.j * (float)ima.getHeight()/(float)WindowDims.h();
      location.i = (int)newi;
      location.j = (int)newj;


      //we got a click, show the featuers at that point
      Rectangle r(Point2D<int>(location.i, location.j), Dims(50, 50));
      track_img = crop(ima, r);
      rutz::shared_ptr<VisualObject> vo(new VisualObject("Track", "TracK", track_img));
      //rutz::shared_ptr<VisualObject> vo(new VisualObject("Track", "TracK", ima));

      vdb.addObject(vo); //add the object to the db
      key_points = vo->getKeypointImage();
      find_obj = true;

    }

    Image< PixRGB<byte> > mimg;
    std::vector<Point2D<int> > tl, tr, br, bl;

    if (find_obj){
      std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
      rutz::shared_ptr<VisualObject> vo(new VisualObject("PIC", "PIC", ima));

      const uint nmatches = vdb.getObjectMatches(vo, matches, VOMA_KDTREEBBF);

      printf("Found %i\n", nmatches);
      if (nmatches > 0 ){
        for(unsigned int i=0; i< nmatches; i++){
          rutz::shared_ptr<VisualObjectMatch> vom = matches[i];
          rutz::shared_ptr<VisualObject> obj = vom->getVoTest();
          LINFO("### Object match with '%s' score=%f",
                obj->getName().c_str(), vom->getScore());

          mimg = vom->getTransfTestImage(mimg);

          // also keep track of the corners of the test image, for
          // later drawing:
          Point2D<int> ptl, ptr, pbr, pbl;
          vom->getTransfTestOutline(ptl, ptr, pbr, pbl);
          tl.push_back(ptl); tr.push_back(ptr);
          br.push_back(pbr); bl.push_back(pbl);

          // do a final mix between given image and matches:
          mimg = Image<PixRGB<byte> >(mimg * 0.5F + ima * 0.5F);

          // finally draw all the object outlines:
          PixRGB<byte> col(255, 255, 0);
          for (unsigned int i = 0; i < tl.size(); i ++)
            {
              drawLine(mimg, tl[i], tr[i], col, 1);
              drawLine(mimg, tr[i], br[i], col, 1);
              drawLine(mimg, br[i], bl[i], col, 1);
              drawLine(mimg, bl[i], tl[i], col, 1);
            }

          window1.drawImage(rescale(mimg, 255, 255));
        }
      } else {
        window1.drawImage(key_points);
      }





    }


    //drawDisk(ima, newwin.p, 10, PixRGB<byte>(255, 0, 0));

    window.drawImage(rescale(ima, 256, 256));



    time += 0.1;
  }



  // got interrupted; let's cleanup and exit:
  manager.stop();
  return 0;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
