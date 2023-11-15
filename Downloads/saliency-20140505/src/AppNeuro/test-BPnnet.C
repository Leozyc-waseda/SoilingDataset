/*!@file AppNeuro/test-BPnnet.C Test BPnnet class */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/test-BPnnet.C $
// $Id: test-BPnnet.C 10982 2009-03-05 05:11:22Z itti $
//

#include "Channels/ChannelOpts.H"
#include "Channels/Jet.H"
#include "Channels/JetFiller.H"
#include "Component/ModelManager.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Media/MediaSimEvents.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/NeuroSimEvents.H"
#include "Neuro/SimulationViewerStd.H"
#include "Neuro/StdBrain.H"
#include "Neuro/VisualCortex.H"
#include "Raster/Raster.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "BPnnet/BPnnet.H"

#include <fstream>
#include <iostream>

//! number of hidden units
#define NHIDDEN 64

//! number of negative samples to generate
#define NNEG 0 /*5*/

//! sampling range in pixels for positive examples around most salient location
#define SRANGE 3

//! sampling step for positive examples, in pixels
#define SSTEP 3

int main(const int argc, const char** argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Attention Model");

  // either run train, recog, or both:
  bool do_jet = false, do_train = false, do_reco = false, do_coords = false;
  if (argc > 1)
    {
      if (strcmp(argv[1], "train") == 0) do_train = true;
      else if (strcmp(argv[1], "reco") == 0) do_reco = true;
      else if (strcmp(argv[1], "coords") == 0) do_coords = true;
      else if (strcmp(argv[1], "jet") == 0) do_jet = true;
      else { LERROR("Incorrect argument(s).  See USAGE."); return 1; }
    }
  else
    {
      LERROR("USAGE:\n  %s jet <label> <img.ppm> <x> <y>\n"
             "  %s train <param> <jetfile> <eta>\n"
             "  %s reco <param> <img.ppm>\n"
             "  %s coords <targetmap.pgm> <img.ppm> <label>\n"
             "where <param> is the stem for parameter files.",
             argv[0], argv[0], argv[0], argv[0]);
      return 1;
    }

  initRandomNumbers();

  // Read input image from disk:
  Image< PixRGB<byte> > col_image;
  if (do_jet || do_reco || do_coords)
    col_image = Raster::ReadRGB(argv[3]);

  // create brain:
  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);
  nub::soft_ref<StdBrain> brain(new StdBrain(manager));
  manager.addSubComponent(brain);
  manager.exportOptions(MC_RECURSE);
  manager.setOptionValString(&OPT_RawVisualCortexChans, "IOC");
  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  // get model started:
  manager.start();
  const uint nborient =
    manager.getModelParamVal<uint>("NumOrientations", MC_RECURSE);

  // build custom JetSpec for our Jets:
  JetSpec *jj = new JetSpec; int jlev = 2, jdepth = 5;
  jj->addIndexRange(RG, RAW, jlev, jlev + jdepth - 1);
  jj->addIndexRange(BY, RAW, jlev, jlev + jdepth - 1);
  jj->addIndexRange(INTENS, RAW, jlev, jlev + jdepth - 1);
  jj->addIndexRange(ORI, RAW, 0, nborient - 1);  // orientations
  jj->addIndexRange(ORI, RAW, jlev, jlev + jdepth - 1);
  rutz::shared_ptr<JetSpec> js(jj);

  // initialize a Jet according to our JetSpec:
  Jet<float> j(js);

  if (do_jet || do_reco || do_coords)
    {
      rutz::shared_ptr<SimEventInputFrame>
        e(new SimEventInputFrame(brain.get(), GenericFrame(col_image), 0));
      seq->post(e); // post the image to the brain
    }

  // do we want to extract a jet?
  if (do_jet)
    {
      // get a jet at specified coordinates:

      LFATAL("fixme");
      /*
      Point2D<int> p(atoi(argv[4]), atoi(argv[5]));
      JetFiller f(p, j, true);
      brain->getVC()->accept(f);
      std::cout<<argv[2]<<' '<<j<<std::endl;
      */
      return 0;
    }

  // do we just want to extract a bunch of coordinates from a target mask?
  if (do_coords)
    {
      Image<byte> tmap = Raster::ReadGray(argv[2]);
      PixRGB<byte> yellowPix(255, 255, 0), greenPix(0, 255, 0);

      // inflate the target mask a bit:
      Image<byte> blownup = chamfer34(tmap);
      blownup = binaryReverse(blownup, byte(255));
      blownup -= 240; blownup *= 255;  // exploit automatic range clamping
      Image<float> sm;

      // mask saliency map by objectmask:
      LFATAL("fixme");
      ///////////      sm = rescale(brain->getVC()->getOutput(), col_image.getDims()) * blownup;

      // find location of most salient point within target mask
      Point2D<int> p; float mval;
      findMax(sm, p, mval);

      LINFO("===== Max Saliency %g at (%d, %d) =====", mval/120.0f, p.i, p.j);
      for (int jj = -SRANGE; jj <= SRANGE; jj += SSTEP)
        for (int ii = -SRANGE; ii <= SRANGE; ii += SSTEP)
          {
            Point2D<int> pp;
            pp.i = std::max(0, std::min(col_image.getWidth()-1, p.i + ii));
            pp.j = std::max(0, std::min(col_image.getHeight()-1, p.j + jj));
            LINFO("===== Positive Sample at (%d, %d) =====", pp.i, pp.j);
            std::cout<<argv[4]<<' '<<argv[3]<<' '<<pp.i<<' '<<pp.j<<std::endl;
          }
      drawPatch(col_image, p, 3, yellowPix);
      drawCircle(col_image, p, 40, yellowPix, 2);

      // now generate a bunch of negative samples, outside target area:
      LFATAL("fixme");
      //////  sm = rescale(brain->getVC()->getOutput(), col_image.getDims()) * binaryReverse(blownup, byte(120));

      for (int i = 0; i < NNEG; i ++)
        {
          findMax(sm, p, mval);

          LINFO("===== Negative Sample at (%d, %d) =====", p.i, p.j);
          std::cout<<"unknown "<<argv[3]<<' '<<p.i<<' '<<p.j<<std::endl;

          drawDisk(sm, p, std::max(col_image.getWidth(),
                                   col_image.getHeight()) / 12, 0.0f);
          drawPatch(col_image, p, 3, greenPix);
          drawCircle(col_image, p, 40, greenPix, 2);
        }

      //Raster::Visu(col_image, "samples.pnm");
      //std::cerr<<"<<<< press [RETURN] to exit >>>"<<std::endl;
      //getchar();

      return 0;
    }

  // read in the knowledge base:
  KnowledgeBase kb; char kn[256]; strcpy(kn, argv[2]); strcat(kn, "_kb.txt");
  kb.load(kn);

  // Create BPnnet and load from disk
  int numHidden = NHIDDEN; // arbitrary for testing
  BPnnet net(js->getJetSize(), numHidden, &kb);
  if (net.load(argv[2]) == false) net.randomizeWeights();

  if (do_train)
    {
      std::ifstream s(argv[3]);
      if (s.is_open() == false)  LFATAL("Cannot read %s", argv[3]);
      double eta = atof(argv[4]);
      char buf[256]; double rms = 0.0; int nb = 0;
      while(!s.eof())
         {
           s.get(buf, 256, ' ');
           if (strlen(buf) > 1)
             {
               SimpleVisualObject vo(buf);
               s>>j; s.getline(buf, 256);

               rms += net.train(j, vo, eta); nb ++;

               net.normalizeWeights();
             }
         }
      s.close();
      rms = sqrt(rms / (double)nb);
      LINFO("Trained %d jets, eta=%.10f: RMS=%.10f", nb, eta, rms);
      net.save(argv[2]);

      std::cout<<rms<<std::endl;
      return 0;
    }

  if (do_reco)
    {
      bool keep_going = true;
      while(keep_going)
        {
          (void) seq->evolve();


          if (SeC<SimEventWTAwinner> e = seq->check<SimEventWTAwinner>(0))
            {
              const Point2D<int> winner = e->winner().p;

              LINFO("##### Winner (%d,%d) at %fms #####",
                    winner.i, winner.j, seq->now().msecs());
              Image< PixRGB<byte> > ctmp;/////////////////FIXME = brain->getSV()->getTraj(seq->now());
              Raster::VisuRGB(ctmp, sformat("traj_%s.ppm", argv[3]));

              LFATAL("fixme");

              /////////              JetFiller f(winner, j, true);
              ////////              brain->getVC()->accept(f);

              SimpleVisualObject vo;
              if (net.recognize(j, vo))
                LINFO("##### Recognized: %s #####", vo.getName());
              else
                LINFO("##### Not Recognized #####");

              std::cout<<"<<<< press [RETURN] to continue >>>"<<std::endl;
              getchar();
              if (seq->now().secs() > 3.0)
                { LINFO("##### Time limit reached #####"); keep_going = false;}
            }
        }
    }

  return 0;
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
