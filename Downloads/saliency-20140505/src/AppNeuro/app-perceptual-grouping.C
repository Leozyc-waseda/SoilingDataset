/*!@file AppNeuro/app-perceptual-grouping.C  Generates perceptual grouping of features
 */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/app-perceptual-grouping.C $
// $Id: app-perceptual-grouping.C 10827 2009-02-11 09:40:02Z itti $
//

#include "Channels/ChannelVisitor.H"
#include "Channels/ComplexChannel.H"
#include "Channels/SingleChannel.H"
#include "Component/ModelManager.H"
#include "Media/FrameSeries.H"
#include "Image/ColorOps.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Media/MediaSimEvents.H"
#include "Neuro/StdBrain.H"
#include "Neuro/NeuroSimEvents.H"
#include "Neuro/VisualCortex.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Raster/Raster.H"
#include "Util/Types.H"

namespace
{
  //! Get the coefficients (gain factors) for perceptual feature grouping
  class CoeffGetter : public ChannelVisitor
  {
    std::vector<double>& itsCoeff;

  public:
    CoeffGetter(std::vector<double>& c) : itsCoeff(c) {}

    virtual ~CoeffGetter() {}

    virtual void visitChannelBase(ChannelBase& chan)
    {
      LFATAL("don't know how to handle %s", chan.tagName().c_str());
    }

    virtual void visitSingleChannel(SingleChannel& chan)
    {
      for (uint idx = 0; idx < chan.numSubmaps(); idx ++)
        {
          // get center and surround scales for this submap index:
          uint clev = 0, slev = 0;
          chan.getLevelSpec().indexToCS(idx, clev, slev);
          // find the coefficient for this submap as a function of the
          // amount of signal present
          double sum = 0.0;
          Image<float> submap = chan.getSubmap(idx);
          float min = 0.0f, max = 0.0f, avg = 0.0f;
          getMinMaxAvg(submap, min, max, avg);
          /*uint w = submap.getWidth(), h = submap.getHeight();
            for (uint i = 0; i < w; i++)
            for (uint j = 0; j < h; j++){
            double salience = submap.getVal(i,j);
            sum += salience*salience*salience;
            }
          */
          sum = max - avg;
          LINFO("%s(%d,%d): -- amount of signal = %lf",
                chan.tagName().c_str(), clev, slev, sum);
          itsCoeff.push_back(sum);
        }
    }

    virtual void visitComplexChannel(ComplexChannel& chan)
    {
      for (uint i = 0; i < chan.numChans(); i++)
        chan.subChan(i)->accept(*this);
    }
  };

  //! Set the coefficients (gain factors) for perceptual feature grouping
  class CoeffSetter : public ChannelVisitor
  {
    const std::vector<double>& itsCoeff;
    uint itsIndex;

  public:
    CoeffSetter(const std::vector<double>& c) : itsCoeff(c), itsIndex(0) {}

    virtual ~CoeffSetter() {}

    virtual void visitChannelBase(ChannelBase& chan)
    {
      LFATAL("don't know how to handle %s", chan.tagName().c_str());
    }

    virtual void visitSingleChannel(SingleChannel& chan)
    {
      const uint num = chan.numSubmaps();
      for (uint i = 0; i < num; ++i)
        {
          uint clev = 0, slev = 0;
          chan.getLevelSpec().indexToCS(i, clev, slev);
          LFATAL("FIXME");
          /////chan.setCoeff(clev, slev, itsCoeff[itsIndex]);
          ++itsIndex;
        }
    }

    virtual void visitComplexChannel(ComplexChannel& chan)
    {
      for (uint i = 0; i < chan.numChans(); ++i)
        chan.subChan(i)->accept(*this);
    }
  };

  //! Compute the percept by grouping features
  class PerceptualGrouping : public ChannelVisitor
  {
    Image<float> itsPercept;

  public:
    PerceptualGrouping() {}

    virtual ~PerceptualGrouping() {}

    Image<float> getPercept() const { return itsPercept; }

    virtual void visitChannelBase(ChannelBase& chan)
    {
      LFATAL("don't know how to handle %s", chan.tagName().c_str());
    }

    virtual void visitSingleChannel(SingleChannel& chan)
    {
      ASSERT(itsPercept.initialized() == false);

      itsPercept = Image<float>(chan.getMapDims(), ZEROS);

      // compute a weighted sum of raw feature maps at all levels:
      for (uint idx = 0; idx < chan.getLevelSpec().maxIndex(); ++idx)
        {
          LFATAL("FIXME");
          const float w = 0.0;////////float(chan.getCoeff(idx));     // weight for that submap
          if (w != 0.0f)
            {
              Image<float> submap = chan.getRawCSmap(idx); // get raw map
              if (w != 1.0f) submap *= w;            // weigh the submap
              // resize submap to fixed scale if necessary:
              if (submap.getWidth() > chan.getMapDims().w())
                submap = downSize(submap, chan.getMapDims());
              else if (submap.getWidth() < chan.getMapDims().w())
                submap = rescale(submap, chan.getMapDims());
              itsPercept += submap;                  // add submap to our sum
            }
        }
    }

    virtual void visitComplexChannel(ComplexChannel& chan)
    {
      ASSERT(itsPercept.initialized() == false);

      itsPercept = Image<float>(chan.getMapDims(), ZEROS);

      for (uint i = 0; i < chan.numChans(); ++i)
        {
          if (chan.getSubchanTotalWeight(i) == 0.0) continue;
          if (chan.subChan(i)->outputAvailable() == false) continue;
          PerceptualGrouping g;
          chan.subChan(i)->accept(g);
          Image<float> subChanOut = g.getPercept();
          const float w = float(chan.getSubchanTotalWeight(i));
          if (w != 1.0f) subChanOut *= w;
          LINFO("%s grouping weight %f",
                chan.subChan(i)->tagName().c_str(), w);
          itsPercept += downSizeClean(subChanOut, itsPercept.getDims());
        }
    }
  };

}

int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Attention Model");

  // Instantiate our various ModelComponents:
  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::soft_ref<StdBrain> brain(new StdBrain(manager));
  manager.addSubComponent(brain);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);

  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  // let's get all our ModelComponent instances started:
  manager.start();

  // main loop:
  bool first=true; std::vector<double> coeff; int count = 0;
                // for perceptual grouping
  while(1) {

    // read new image in?
    const FrameState is = ifs->update(seq->now());
    if (is == FRAME_COMPLETE) break; // done
    if (is == FRAME_NEXT || is == FRAME_FINAL) // new frame
      {
        Image< PixRGB<byte> > input = ifs->readRGB();

        // empty image signifies end-of-stream
        if (input.initialized())
          {
            rutz::shared_ptr<SimEventInputFrame>
              e(new SimEventInputFrame(brain.get(), GenericFrame(input), 0));
            seq->post(e); // post the image to the brain

            // show memory usage if in debug mode:
            if (MYLOGVERB >= LOG_DEBUG)
              SHOWMEMORY("MEMORY USAGE: frame %d t=%.1fms", ifs->frame(),
                         seq->now().msecs());
          }
      }

    // evolve brain:
    seq->evolve();


    // write outputs or quit?
    bool gotcovert = false;
    if (seq->check<SimEventWTAwinner>(0)) gotcovert = true;
    const FrameState os = ofs->update(seq->now(), gotcovert);

    if (os == FRAME_NEXT || os == FRAME_FINAL) // new FOA
      {
        brain->save(SimModuleSaveInfo(ofs, *seq));

        // arbitrary: every time we have a winner, change the percept
        // get the gain factors based on amount of signal present
        // within the feature map
        LINFO ("perceptual feature grouping");
        if (first) {
          LINFO ("first iteration...getting coefficients");

          LFATAL("fixme");

          //////          CoeffGetter g(coeff);
          /////// brain->getVC()->accept(g);
          first = false;
        }
        LINFO ("obtained the coefficients...");
        // for perceptual feature grouping, iterate through the gain
        // factors, setting one of them to 1.0 and the rest to 0.0 in
        // decsending order
        uint idx = 0; // idx corresponding to the max
        for(uint i = 1; i < coeff.size(); i++)
          if (coeff[i] > coeff[idx])
            idx = i;
        std::vector<double> perceptCoeff;
        for(uint i = 0; i < coeff.size(); i++)
          perceptCoeff.push_back(0.0);
        perceptCoeff[idx] = 1.0;
        {
          LFATAL("fixme");
          ////////CoeffSetter s(perceptCoeff);
          // set the new coefficients on the dummy VC
          ///////brain->getVC()->accept(s);
        }
        LINFO ("iteration %d: gain of chan %d = 1.0", count+1, idx);
        // prevent current max from being chosen again
        coeff[idx] = -1.0; count++;

        // compute the percept
        LFATAL("fixme");
        PerceptualGrouping pg;
        ///////brain->getVC()->accept(pg);
        Image<float> percept = pg.getPercept();
        char out[10];
        sprintf(out,"%d_chan%d",count,idx);
        //Raster::WriteFloat(percept, FLOAT_NORM_0_255, sformat("percept_%s.pgm",out));
        inplaceNormalize(percept, 0.0f, 1.0f);
        Image< PixRGB<byte> > input = ifs->readRGB();
        percept = rescale(percept, input.getDims());
        input = input * percept;
        normalizeC (input, 0, 255);
        Raster::WriteRGB (input, sformat("percept_%s.ppm",out));

      }

    if (os == FRAME_FINAL)
      break;

    // if we displayed a bunch of images, let's pause:
    if (ifs->shouldWait() || ofs->shouldWait())
      Raster::waitForKey();
  }

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
