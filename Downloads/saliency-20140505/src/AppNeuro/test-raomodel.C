/*!@file AppNeuro/test-raomodel.C test model by Rao et al., Vis Res 2002 */

// //////////////////////////////////////////////////////////////////// //
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
//
// Primary maintainer for this file: Philip Williams <plw@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/test-raomodel.C $
// $Id: test-raomodel.C 10982 2009-03-05 05:11:22Z itti $
//

#include "Channels/ChannelBase.H"
#include "Channels/Jet.H"
#include "Channels/JetFiller.H"
#include "Component/ModelManager.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Media/MediaSimEvents.H"
#include "Channels/ChannelOpts.H"
#include "Channels/RawVisualCortex.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Simulation/SimEventQueue.H"
#include "Raster/Raster.H"
#include "Util/log.H"

#include <fstream>
#include <string>

#define SMIN 1
#define SMAX 3

namespace
{
  // ######################################################################
  Image<float> getRaoJetMap(RawVisualCortex& vc,
                            const Jet<float>& targ,
                            const int smin, const int smax)
  {
    const int w = vc.getInputDims().w() >> smin;
    const int h = vc.getInputDims().h() >> smin;
    Image<float> result(w, h, NO_INIT);
    Jet<float> currJet(targ.getSpec());

    // now loop over the entire image and compute a weighted Jet
    // distance between that central Jet and the Jets extracted from all
    // the other locations in the image:
    for (int x = 0; x < w; x++)
      for (int y = 0; y < h; y++)
        {
          JetFiller f(Point2D<int>(x << smin, y << smin), currJet, false);
          vc.accept(f);
          result.setVal(x, y, float(raodistance(targ, currJet, smin, smax)));
        }
    return result;
  }
}

int main(const int argc, const char** argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Rao Model");

  // create brain:
  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::soft_ref<RawVisualCortex> vcx(new RawVisualCortex(manager));
  manager.addSubComponent(vcx);
  manager.setOptionValString(&OPT_RawVisualCortexChans, "ICO");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "jet <image.ppm> <x> <y> <jet.txt>   -OR-   "
                               "jet2 <image.ppm> <target.pgm> <jet.txt> -OR-  "
                               "snr <image.ppm> <target.pgm> <jet.txt> <lambda> <map.pgm>  -OR-  "
                               "search <image.ppm> <jet.txt> <lambda>",
                               4, 6) == false)
    return(1);

  // make sure all feature values are in [0..255]:
  manager.setModelParamVal("GaborChannelIntensity", 100.0, MC_RECURSE);
  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  bool do_jet = false, do_jet2 = false, do_snr = false, do_search = false;
  std::string action = manager.getExtraArg(0);
  if (action.compare("jet") == 0)
    {
      if (manager.numExtraArgs() != 5)
        LFATAL("USAGE: %s jet <image.ppm> <x> <y> <jet.txt>", argv[0]);
      do_jet = true;
    }
  else if (action.compare("jet2") == 0)
    {
      if (manager.numExtraArgs() != 4)
        LFATAL("USAGE: %s jet2 <image.ppm> <target.pgm> <jet.txt>", argv[0]);
      do_jet2 = true;
    }
  else if (action.compare("snr") == 0)
    {
      if (manager.numExtraArgs() != 6)
        LFATAL("USAGE: %s snr <image.ppm> <target.pgm> <jet.txt> <lambda> <map.pgm>", argv[0]);
      do_snr = true;
    }
  else if (action.compare("search") == 0)
    {
      if (manager.numExtraArgs() != 4)
        LFATAL("USAGE: %s search <image.ppm> <jet.txt> <lambda>", argv[0]);
      do_search = true;
    }
  else
    LFATAL("Incorrect usage -- try to run without args to see usage.");

  // Read input image from disk:
  Image< PixRGB<byte> > col_image =
    Raster::ReadRGB(manager.getExtraArg(1));

  // get model started:
  manager.start();

  // process the input image:
  vcx->input(InputFrame::fromRgb(&col_image));
  vcx->getOutput();

  rutz::shared_ptr<JetSpec> js(new JetSpec);


  js->addIndexRange(COLBAND, RAW, 0, 5);
  js->addIndexRange(COLBAND, RAW, SMIN, SMAX);
  js->addIndexRange(INTENS, RAW, 0, 3);
  js->addIndexRange(INTENS, RAW, SMIN, SMAX);
  /*
  js->addIndexRange(RG, RAW, SMIN, SMAX);
  js->addIndexRange(BY, RAW, SMIN, SMAX);
  js->addIndexRange(INTENS, RAW, SMIN, SMAX);
  */


  const uint nori = vcx->subChan("orientation")->getModelParamVal<uint>("NumOrientations");
  js->addIndexRange(ORI, RAW, 0, nori - 1);
  js->addIndexRange(ORI, RAW, SMIN, SMAX);
  js->print();

  // ===================================================================
  // do we want to just read out a jet using the (x,y) coordinate?
  if (do_jet)
    {
      int x = manager.getExtraArgAs<int>(2);
      int y = manager.getExtraArgAs<int>(3);
      Point2D<int> p(x, y);

      // initialize a Jet according to our JetSpec:
      Jet<float> j(js);
      JetFiller f(p, j, false);
      vcx->accept(f);

      // save it to disk:
      std::ofstream s(manager.getExtraArg(4).c_str());
      if (s.is_open() == false)
        LFATAL("Cannot write %s", manager.getExtraArg(4).c_str());
      s<<j<<std::endl;
      s.close();
      LINFO("Saved Jet(%d, %d) to %s -- DONE.", x, y,
            manager.getExtraArg(4).c_str());
    }

  // =========================================================================
  // do we want to just read out a jet using the target mask?
  if (do_jet2)
    {
      // Read target image from disk:
      Image<byte> target =
        Raster::ReadGray(manager.getExtraArg(2));

      // find the target coordinates
      int w = target.getWidth(), h = target.getHeight();
      int t = h+1, b = -1, l = w+1, r = -1;
      for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++)
          if (target.getVal(i,j) > 200){
            if (j < t) t = j;
            if (i < l) l = i;
            if (i > r) r = i;
            if (j > b) b = j;
          }
      int x = (r + l) / 2;
      int y = (t + b)/ 2;
      Point2D<int> p(x, y);

      // initialize a Jet according to our JetSpec:
      Jet<float> j(js);
      JetFiller f(p, j, false);
      vcx->accept(f);

      // save it to disk:
      std::ofstream s(manager.getExtraArg(3).c_str());
      if (s.is_open() == false)
        LFATAL("Cannot write %s", manager.getExtraArg(3).c_str());
      s<<j<<std::endl;
      s.close();
      LINFO("Saved Jet(%d, %d) to %s -- DONE.", x, y,
            manager.getExtraArg(3).c_str());
    }

  // =======================================================================
  // do we want to get a jetmap?
  if (do_snr)
    {
      // Read target image from disk:
      Image<byte> targetMask =
        Raster::ReadGray(manager.getExtraArg(2));

      std::ifstream s(manager.getExtraArg(3).c_str());
      if (s.is_open() == false)
        LFATAL("Cannot read %s", manager.getExtraArg(3).c_str());
      Jet<float> j(js);
      s>>j; s.close();

      float lambda = manager.getExtraArgAs<float>(4);

      // get the coarsest map:
      Image<float> jmap = getRaoJetMap(*vcx, j, SMAX, SMAX);
      float mi, ma; getMinMax(jmap, mi, ma);
      LINFO("jmap range [%f .. %f]", mi, ma);

      // apply softmax: Note that here we also divide the squared
      // distance values by 65535 such as to bring them back to a
      // [0..1] range which I guess was assumed by Rao et al:
      Image<float> fmap = exp(jmap * (-1.0f / (lambda*65535.0f)));
      float denom = sum(fmap); fmap *= 1.0f / denom;

      // compute SNR = max(sT) / max(sD)
      Image<float> targetMap(fmap.getDims(), ZEROS);
      float sT = 0.0f, sD = 0.0f;
      float BG_FIRING_RATE = 0.1f;
      if (targetMask.initialized()){
        // first scale the target mask to match the size of fmap
        if (targetMask.getWidth() > fmap.getWidth())
          targetMap = downSize(targetMask, fmap.getDims());
        else if (targetMask.getWidth() < fmap.getWidth())
          targetMap = rescale(targetMask, fmap.getDims());
        else
          targetMap = targetMask;
        // find sT as max salience within the target object
        // find sD as max salience outside the target object
        Image<float>::const_iterator aptr = targetMap.begin(),
          astop = targetMap.end();
        Image<float>::const_iterator sptr = fmap.begin(),
          sstop = fmap.end();
        while (aptr != astop && sptr != sstop)
          {
            if (*aptr > 0.0f){
              if (sT < *sptr)
                sT = *sptr;
            }
            else {
              if (sD < *sptr)
                sD = *sptr;
            }
            aptr++; sptr++;
          }
      }
      // find SNR
      float SNR = log((sT + BG_FIRING_RATE) / (sD + BG_FIRING_RATE));
      LINFO ("sT = %f, sD = %f", sT, sD);
      LINFO ("-------------- SNR = %f dB", SNR);

      // output the SNR in a file
      FILE * fout = fopen("snr", "w");
      fprintf(fout, " SNR = %f dB", SNR);
      fclose(fout);

      // save the map:
      Raster::WriteFloat(fmap, FLOAT_NORM_0_255, manager.getExtraArg(5), RASFMT_PNM);

      LINFO("Saved Fmap to %s -- DONE.", manager.getExtraArg(5).c_str());
    }

  // ======================================================================
  // do we want to search?
  if (do_search)
    {
      std::ifstream s(manager.getExtraArg(2).c_str());
      if (s.is_open() == false)
        LFATAL("Cannot read %s", manager.getExtraArg(2).c_str());
      Jet<float> j(js);
      s>>j; s.close();

      // start with the coarsest scale only and progressively include
      // more scales:
      float lambda = atof(manager.getExtraArg(3).c_str());
      for (int k = SMAX; k >= SMIN; k --)
        {
          LINFO("Using scales [%d .. %d], lambda = %f", k, SMAX, lambda);

          // get the map:
          Image<float> jmap = getRaoJetMap(*vcx, j, k, SMAX);
          float mi, ma; getMinMax(jmap, mi, ma);
          LINFO("jmap range [%f .. %f]", mi, ma);

          // apply softmax: Note that here we also divide the squared
          // distance values by 65535 such as to bring them back to a
          // [0..1] range which I guess was assumed by Rao et al:
          Image<float> fmap = exp(jmap * (-1.0f / (lambda*65535.0f)));
          float denom = sum(fmap); fmap *= 1.0f / denom;

          // find the saccade target:
          float xhat = 0.0f, yhat = 0.0f;
          int w = fmap.getWidth(), h = fmap.getHeight();
          for (int jj = 0; jj < h; jj ++)
            for (int ii = 0; ii < w; ii ++)
              {
                float val = fmap.getVal(ii, jj);
                xhat += ii * val;
                yhat += jj * val;
              }
          getMinMax(fmap, mi, ma);
          LINFO("fmap range [%f .. %f]", mi, ma);
          Point2D<int> win(int(xhat + 0.499f) << k, int(yhat + 0.499f) << k);
          LINFO("Saccade to (%d, %d)", win.i, win.j);

          // display fmap and winner:
          Raster::VisuFloat(fmap, FLOAT_NORM_0_255, "fmap.pgm");
          Image< PixRGB<byte> > traj(col_image);
          drawPatch(traj, win, 5, PixRGB<byte>(255, 255, 0));
          int foar = std::max(traj.getWidth(), traj.getHeight()) / 12;
          drawCircle(traj, win, foar, PixRGB<byte>(255, 255, 0), 3);
          Raster::VisuRGB(traj, "traj.ppm");


          // get ready for next saccade:
          lambda *= 0.5f;
        }
    }

  // all done!
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
