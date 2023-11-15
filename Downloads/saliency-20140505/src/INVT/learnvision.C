/*!@file INVT/learnvision.C  like ezvision.C but focused on learning */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/learnvision.C $
// $Id: learnvision.C 12074 2009-11-24 07:51:51Z itti $
//

#include "Channels/ChannelBase.H"
#include "Channels/ChannelVisitor.H"
#include "Channels/SingleChannel.H"
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H" // for rescale()
#include "Image/Transforms.H" // for chamfer34()
#include "Channels/RawVisualCortex.H"
#include "Raster/Raster.H"
#include "Util/SimTime.H"
#include "Util/Types.H"

#include <vector>
#include <cstdio>

namespace
{

  class CoeffLearner : public ChannelVisitor
  {
  public:
    CoeffLearner(const Image<byte>& dmap, const double eta,
                 const bool softmask,
                 const int inthresh, const int outthresh)
      :
      itsDmap(dmap),
      itsEta(eta),
      itsSoftmask(softmask),
      itsInThresh(inthresh),
      itsOutThresh(outthresh),
      itsAbsSumCoeffs()
    {
      itsAbsSumCoeffs.push_back(0.0);
    }

    virtual ~CoeffLearner() {}

    double absSumCoeffs() const
    {
      ASSERT(itsAbsSumCoeffs.size() == 1);
      return itsAbsSumCoeffs.back();
    }

    virtual void visitChannelBase(ChannelBase& chan)
    {
      LFATAL("don't know how to handle %s", chan.tagName().c_str());
    }

    virtual void visitSingleChannel(SingleChannel& chan)
    {
      if (chan.visualFeature() == FLICKER)
        {
          // do nothing; we can't "learn" flicker from a single input
          // image
          return;
        }

      chan.killCaches();
      /* FIXME
      const LevelSpec ls = chan.getLevelSpec();

      for (uint del = ls.delMin(); del <= ls.delMax(); ++del)
        for (uint lev = ls.levMin(); lev <= ls.levMax(); ++lev)
          {
            const uint idx = ls.csToIndex(lev, lev+del);
            const Image<float> fmap = chan.getSubmap(idx);
            const double oldcoeff = chan.getCoeff(idx);

            const double newcoeff = oldcoeff +
              itsEta * learningCoeff(fmap,
                                     rescale(itsDmap, fmap.getDims()),
                                     itsSoftmask,
                                     itsInThresh, itsOutThresh);

            chan.setCoeff(idx, newcoeff);

            LINFO("%s(%d,%d): %f -> %f",
                  chan.tagName().c_str(), lev, lev+del,
                  oldcoeff, newcoeff);
          }

      chan.clampCoeffs(0.0, 100.0);

      ASSERT(itsAbsSumCoeffs.size() > 0);
      itsAbsSumCoeffs.back() += chan.absSumCoeffs();
      */
    }

    virtual void visitComplexChannel(ComplexChannel& chan)
    {
      chan.killCaches();
      /* FIXME
      for (uint i = 0; i < chan.numChans(); ++i)
        {
          itsAbsSumCoeffs.push_back(0.0);
          chan.subChan(i)->accept(*this);
          const double wt =
            clampValue(chan.getSubchanTotalWeight(i) / chan.numSubmaps(),
                       0.0, 100.0);
          chan.setSubchanTotalWeight(i, wt * chan.numSubmaps());
          itsAbsSumCoeffs.back() *= chan.getSubchanTotalWeight(i);

          const double subsum = itsAbsSumCoeffs.back();
          itsAbsSumCoeffs.pop_back();

          itsAbsSumCoeffs.back() += subsum;
        }
      */
      // I leave to the user the opportunity to normalize the coeffs
      // or not after each learning. Some normalization should be done
      // at some point to prevent coeff blowout.
    }

  private:
    const Image<byte> itsDmap;
    const double itsEta;
    const bool itsSoftmask;
    const int itsInThresh;
    const int itsOutThresh;

    std::vector<double> itsAbsSumCoeffs;
  };


  class CoeffNormalizer : public ChannelVisitor
  {
  public:
    CoeffNormalizer(const double div)
      :
      itsDiv(div)
    {}

    virtual ~CoeffNormalizer() {}

    virtual void visitChannelBase(ChannelBase& chan)
    {
      LFATAL("don't know how to handle %s", chan.tagName().c_str());
    }

    virtual void visitSingleChannel(SingleChannel& chan)
    {
      ////FIXME      chan.normalizeCoeffs(itsDiv);
    }

    virtual void visitComplexChannel(ComplexChannel& chan)
    {
      for (uint i = 0; i < chan.numChans(); ++i)
        chan.subChan(i)->accept(*this);
    }

  private:
    const double itsDiv;
  };

}

//! Basic program to learn feature map weights from static images
/*! This program allows training of the relative weights of feature
  maps, given an image and associated binary target mask. */
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Attention Model");

  // Instantiate our various ModelComponents:
  nub::soft_ref<RawVisualCortex> vcx(new RawVisualCortex(manager));
  manager.addSubComponent(vcx);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<image> <targetMask> <coeffs.pmap> "
                               "<D|N> <inthresh> <outthresh> <eta>",
                               7, 7) == false)
    return(1);

  // do post-command-line configs:
  Image< PixRGB<byte> > image = Raster::ReadRGB(manager.getExtraArg(0));
  Image<byte> targetmask = Raster::ReadGray(manager.getExtraArg(1));

  // let's get all our ModelComponent instances started:
  manager.start();

  // load the weights:
  FILE *f = fopen(manager.getExtraArg(2).c_str(), "r");
  if (f) {
    fclose(f);
    LINFO("Loading params from %s", manager.getExtraArg(2).c_str());
    /////FIXME    vcx->readParamMap(manager.getExtraArg(2).c_str());
  }

  // process the input image:
  vcx->input(InputFrame::fromRgb(&image));

  // learn:
  bool doDistMap = false;
  if (manager.getExtraArg(3).c_str()[0] == 'D') doDistMap = true;
  int inthresh = manager.getExtraArgAs<int>(4);
  int outthresh = manager.getExtraArgAs<int>(5);
  double eta = manager.getExtraArgAs<double>(6);
  const double softmask = true;

  // create a chamfer distance map from the target image -> target
  // weighting; result has zeros inside the targetmask and 255 outside,
  // graded values in between:
  Image<byte> dmap;
  if (doDistMap) dmap = chamfer34(targetmask);
  else dmap = binaryReverse(targetmask, byte(255));

  CoeffLearner l(dmap, eta, softmask, inthresh, outthresh);
  vcx->accept(l);

  const double sum = l.absSumCoeffs();

  if (sum < 0.1)
    {
      LERROR("Sum of coeffs very small (%f). Not normalized.", sum);
    }
  else
    {
      const uint nbmaps = vcx->numSubmaps();
      LINFO("Coeff normalization: old sum = %f, nbmaps = %d",
            sum, nbmaps);

      CoeffNormalizer n(sum / double(nbmaps));
      vcx->accept(n);
    }

  // save the new params:
  LINFO("Saving params to %s", manager.getExtraArg(2).c_str());
  ////FIXME  vcx->writeParamMap(manager.getExtraArg(2).c_str());

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
