/*!@file Apps/BorderWatch/BorderWatch.C Border watch */

// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Apps/BorderWatch/BorderWatch.C $
// $Id: BorderWatch.C 13039 2010-03-23 02:06:32Z itti $

#include "Component/JobServerConfigurator.H"
#include "Component/ModelManager.H"
#include "Component/GlobalOpts.H" // for OPT_UseRandom
#include "Channels/ChannelOpts.H" // for OPT_LevelSpec
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"
#include "Image/Layout.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/XWinManaged.H"
#include "GUI/DebugWin.H"
#include "GUI/SimpleMeter.H"
#include "Neuro/EnvOpts.H" // for OPT_EnvLevelSpec, etc
#include "Neuro/NeuroOpts.H" // for OPT_VisualCortexType
#include "Apps/BorderWatch/ImageInfo.H"
#include "Simulation/SimulationOpts.H"
#include "Simulation/SimEventQueue.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Util/Timer.H"
#include <queue>
#include <cmath>
#include <cstdio>
#include <fstream>

const ModelOptionCateg MOC_BORDERWATCH = {
  MOC_SORTPRI_3, "BorderWatch-Related Options" };

static const ModelOptionDef OPT_ShowDebugWin =
  { MODOPT_FLAG, "ShowDebugWin", &MOC_BORDERWATCH, OPTEXP_CORE,
    "Whether to show the input image, the saliency maps, and the beliefs. "
    "This is used for debugging and setting the thresholds. ",
    "show-debug-win", '\0', "", "false" };

static const ModelOptionDef OPT_ShowThumbs =
  { MODOPT_FLAG, "ShowThumbs", &MOC_BORDERWATCH, OPTEXP_CORE,
    "Show thumbnails of the most recent frames that had surprise above threshold",
    "show-thumbs", '\0', "", "false" };

static const ModelOptionDef OPT_Threshold =
  { MODOPT_ARG(float), "Threshold", &MOC_BORDERWATCH, OPTEXP_CORE,
    "The threshold level at which to save images to disk. ",
    "threshold", '\0', "<float>", "3.5e-10" };

static const ModelOptionDef OPT_ImgQLen =
  { MODOPT_ARG(uint), "ImgQLen", &MOC_BORDERWATCH, OPTEXP_CORE,
    "Length of the queue of images for movie context, in frames",
    "imgqlen", '\0', "<uint>", "50" };

const ModelOptionDef OPT_OutFname =
  { MODOPT_ARG_STRING, "OutFname", &MOC_BORDERWATCH, OPTEXP_CORE,
    "File name for text output data (or empty to not save output data)",
    "out-fname", '\0', "<file>", "" };

// ######################################################################
void displayOutput(const Image<PixRGB<byte> >& img, nub::ref<OutputFrameSeries> ofs, bool showThumbs)
{
  static const uint nx = 15, ny = 12; // hardcoded for 30in display
  static ImageSet<PixRGB<byte> > thumbs(nx*ny, Dims(img.getWidth()/2, img.getHeight()/2), ZEROS);

  // do we want to show thumbnail displays?
  if (showThumbs) {
    thumbs.push_back(quickLocalAvg2x2(img)); while(thumbs.size() > nx*ny) thumbs.pop_front();
    Layout<PixRGB<byte> > t = arrcat(thumbs, nx);
    ofs->writeRgbLayout(t, "Thumbnails", FrameInfo("Thumbnails", SRC_POS));
  }

  // display the full-resolution image:
  ofs->writeRGB(img, "Output", FrameInfo("Output", SRC_POS));
  ofs->updateNext();
}

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;
  ModelManager mgr("Border Watch");

  nub::ref<JobServerConfigurator> jsc(new JobServerConfigurator(mgr));
  mgr.addSubComponent(jsc);

  nub::ref<SimEventQueueConfigurator> seqc(new SimEventQueueConfigurator(mgr));
  mgr.addSubComponent(seqc);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(mgr));
  mgr.addSubComponent(ofs);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(mgr));
  mgr.addSubComponent(ifs);

  nub::ref<ImageInfo> imageInfo(new ImageInfo(mgr));
  mgr.addSubComponent(imageInfo);

  OModelParam<bool> optShowDebugWin(&OPT_ShowDebugWin, &mgr);
  OModelParam<bool> optShowThumbs(&OPT_ShowThumbs, &mgr);
  OModelParam<float> optThreshold(&OPT_Threshold, &mgr);
  OModelParam<uint> optQLen(&OPT_ImgQLen, &mgr);
  OModelParam<std::string> optOutFname(&OPT_OutFname, &mgr);

  mgr.exportOptions(MC_RECURSE);
  mgr.setOptionValString(&OPT_SimulationTimeStep, "30ms"); // avoid feeding inputs too fast
  mgr.setOptionValString(&OPT_LevelSpec, "1,3,3,4,4"); // use higher saliency resolution than usual
  mgr.setOptionValString(&OPT_VisualCortexType, "Env"); // use envision
  mgr.setOptionValString(&OPT_EvcMultithreaded, "true"); // use envision multi-threaded
  mgr.setOptionValString(&OPT_EnvLevelSpec, "1,3,3,4,4"); // use higher saliency resolution than usual
  mgr.setOptionValString(&OPT_UseRandom, "false"); // enough noise in the videos, no need to add more

  if (mgr.parseCommandLine(argc, argv, "", 0, 0) == false)
    { LFATAL("This program takes no non-option command-line argument; try --help"); return 1; }

  nub::ref<SimEventQueue> seq = seqc->getQ();

  // open output file, if any:
  std::ofstream *outFile = 0;
  if (optOutFname.getVal().empty() == false)
    {
      outFile = new std::ofstream(optOutFname.getVal().c_str());
      if (outFile->is_open() == false)
        LFATAL("Cannot open '%s' for writing", optOutFname.getVal().c_str());
    }

  // get started:
  mgr.start();
  MYLOGVERB = LOG_CRIT; // get less verbose...

  std::queue<Image<PixRGB<byte> > > imgQueue;
  SimStatus status = SIM_CONTINUE;
  Timer tim;

  // main loop:
  while(status == SIM_CONTINUE) {
    ifs->updateNext();

    // grab the images:
    GenericFrame input = ifs->readFrame(); if (!input.initialized()) break;
    Image<PixRGB<byte> > img = input.asRgb();

    // process the image and get the results:
    ImageInfo::ImageStats stats = imageInfo->update(seq, img, ifs->frame());

    float mi, ma;
    Image<float> smapf = stats.smap; inplaceNormalize(smapf, 0.0F, 255.0F, mi, ma);
    Image<byte> smap = smapf; smap = rescaleNI(smap, img.getDims());

    // time-stamp the image:
    char msg[255]; time_t rawtime; time(&rawtime);
    struct tm *timeinfo; timeinfo = localtime(&rawtime);
    sprintf(msg, " %s- S=%g %d", asctime(timeinfo), stats.score, ifs->frame());
    writeText(img, Point2D<int>(0,0), msg, PixRGB<byte>(255,0,0), PixRGB<byte>(0), SimpleFont::FIXED(6), true);
    sprintf(msg, " Saliency [%.4g .. %.4g] %d ", mi, ma, ifs->frame());
    writeText(smap, Point2D<int>(0,0), msg, byte(255), byte(0), SimpleFont::FIXED(6));

    // mark most salient point:
    const Point2D<int> sp = stats.salpoint * img.getWidth() / smapf.getWidth();
    if (stats.score > optThreshold.getVal()) drawCircle(img, sp, 15, PixRGB<byte>(255,255,0), 2);
    else drawCircle(img, sp, 5, PixRGB<byte>(0,128,0), 1);

    // push the image into our context queue:
    imgQueue.push(img);
    if (imgQueue.size() > optQLen.getVal()) imgQueue.pop();

    // show a debug window with saliency maps and such?
    if (optShowDebugWin.getVal())
      {
        if (stats.belief1.initialized() && stats.belief2.initialized()) {
          Image<float> b1f = stats.belief1; inplaceNormalize(b1f, 0.0F, 255.0F, mi, ma);
          Image<byte> b1 = b1f; b1 = rescaleNI(b1, img.getDims());
          sprintf(msg, " Beliefs 1 [%.4g .. %.4g] ", mi, ma);
          writeText(b1, Point2D<int>(0,0), msg, byte(255), byte(0), SimpleFont::FIXED(6));
          drawLine(b1, Point2D<int>(0,0), Point2D<int>(0, b1.getHeight()-1), byte(255));

          Image<float> b2f = stats.belief2; inplaceNormalize(b2f, 0.0F, 255.0F, mi, ma);
          Image<byte> b2 = b2f; b2 = rescaleNI(b1, img.getDims());
          sprintf(msg, " Beliefs 2 [%.4g .. %.4g] ", mi, ma);
          writeText(b2, Point2D<int>(0,0), msg, byte(255), byte(0), SimpleFont::FIXED(6));
          drawLine(b2, Point2D<int>(0,0), Point2D<int>(0, b2.getHeight()-1), byte(255));

          Layout<byte> smapDisp;
          smapDisp = hcat(hcat(smap, b1), b2);
          ofs->writeGrayLayout(smapDisp, "Smap", FrameInfo("Smap", SRC_POS));
        } else ofs->writeGray(smap, "Smap", FrameInfo("Smap", SRC_POS));

        const MeterInfo infos[] = {
          { "", stats.score, optThreshold.getVal()*3, optThreshold.getVal(), PixRGB<byte>(0, 255, 0) }
        };
        Image<PixRGB<byte> > meterImg =
          drawMeters(&infos[0], sizeof(infos) / sizeof(infos[0]), 1, Dims(img.getWidth(),20));

        Layout<PixRGB<byte> > inputDisp;
        inputDisp = vcat(img, meterImg);
        ofs->writeRgbLayout(inputDisp, "Input", FrameInfo("Input", SRC_POS));
      }

    if (stats.score > optThreshold.getVal())
      {
        if (imgQueue.size() == optQLen.getVal()) // buffer is full, place a title
          {
            Image<PixRGB<byte> > title = imgQueue.back();

            sprintf(msg, "Date: %s                   ", asctime(timeinfo));
            writeText(title, Point2D<int>(0,title.getHeight()-40), msg, PixRGB<byte>(255), PixRGB<byte>(0));
            sprintf(msg, "Surprise: %e                              ", stats.score);
            writeText(title, Point2D<int>(0,title.getHeight()-20), msg, PixRGB<byte>(255), PixRGB<byte>(0));

            for (int i = 0; i < 3; ++i) displayOutput(title, ofs, optShowThumbs.getVal());
          }

        while(!imgQueue.empty())
          {
            Image<PixRGB<byte> > tImg = imgQueue.front(); imgQueue.pop();
            displayOutput(tImg, ofs, optShowThumbs.getVal());
          }
      }

    // print framerate:
    if (ifs->frame() > 0 && (ifs->frame() % 100) == 0) {
      printf("Frame %06d - %.2ffps\n", ifs->frame(), 100.0 / tim.getSecs());
      tim.reset();
    }

    // save text output if desired, for later use:
    if (outFile)
      (*outFile)<<sformat("of=%i if=%i score=%e x=%d y=%d sal=%e ener=%e uniq=%e entr=%e rand=%e kls=%e msds=%e %s",
			  ofs->frame(), ifs->frame(), stats.score, sp.i, sp.j, stats.saliency, stats.energy,
			  stats.uniqueness, stats.entropy, stats.rand, stats.KLsurprise, stats.MSDsurprise,
			  asctime(timeinfo));

    // Evolve for one time step and switch to the next one:
    status = seq->evolve();
  }

  LINFO("Simulation terminated.");
  mgr.stop();
  if (outFile) { outFile->close(); delete outFile; }

  return 0;
}


