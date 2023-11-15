/*!@file TIGS/TigsInputFrame.H Class that lets us do lazy computation of luminance/rg/by from an input frame */

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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/TigsInputFrame.H $
// $Id: TigsInputFrame.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef TIGS_TIGSINPUTFRAME_H_DEFINED
#define TIGS_TIGSINPUTFRAME_H_DEFINED

#include "Image/Hash.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/SimTime.H"
#include "Util/log.H"
#include "rutz/shared_ptr.h"

/// Class that lets us do lazy computation of luminance/rg/by from an input frame
class TigsInputFrame
{
public:
  TigsInputFrame(const Image<PixRGB<byte> >& in, const SimTime& t)
    :
    itsGhost(false),
    itsTime(t), itsHashInited(false),
    itsOrigbounds(in.getBounds()),
    itsOrigframe(in)
  {
    // obviously hackish -- but for now there are various places in
    // this file where we make the implicit assumption of 640x480
    // input, so let's just verify it at the get-go here:
    ASSERT(in.getDims() == Dims(640,480));
  }

  TigsInputFrame(const SimTime& t, const Dims& dims,
                 const Digest<16>& hash)
    :
    itsGhost(true),
    itsTime(t), itsHashInited(true), itsHash(hash),
    itsOrigbounds(Rectangle(Point2D<int>(0,0), dims))
  {
    ASSERT(dims == Dims(640,480));
  }

  static rutz::shared_ptr<TigsInputFrame>
  fromGhostString(const std::string& s);

  std::string toGhostString() const;

  bool isGhost() const { return itsGhost; }

  SimTime t() const
  { return itsTime; }

  Digest<16> getHash() const
  {
    if (!itsHashInited)
      {
        if (itsGhost)
          LFATAL("ghost frames must have a precomputed hash");

        // let's hash the simtime into the digest as well:
        const int msec = int(itsTime.msecs());

        itsHash = md5rgb(&itsOrigframe, &msec, sizeof(msec));
        itsHashInited = true;
      }

    return itsHash;
  }

  const Rectangle& origbounds() const
  { return itsOrigbounds; }

  const Image<PixRGB<byte> >& origframe() const
  {
    if (itsGhost)
      LFATAL("origframe() not allowed for ghost frames");

    return itsOrigframe;
  }

  const Image<PixRGB<byte> >& rgb() const
  {
    if (itsGhost)
      LFATAL("rgb() not allowed for ghost frames");

    initialize();
    return itsRGB;
  }

  const Image<float>& lum() const
  {
    if (itsGhost)
      LFATAL("lum() not allowed for ghost frames");

    initialize();
    return itsLum;
  }

  const Image<float>& rg() const
  {
    if (itsGhost)
      LFATAL("rg() not allowed for ghost frames");

    initialize();
    return itsRG;
  }

  const Image<float>& by() const
  {
    if (itsGhost)
      LFATAL("by() not allowed for ghost frames");

    initialize();
    return itsBY;
  }

private:
  void initialize() const;

  bool itsGhost;

  SimTime itsTime;

  mutable bool itsHashInited;
  mutable Digest<16> itsHash;

  Rectangle itsOrigbounds;

  Image<PixRGB<byte> > itsOrigframe;
  mutable Image<PixRGB<byte> > itsRGB;
  mutable Image<float> itsLum;
  mutable Image<float> itsRG;
  mutable Image<float> itsBY;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_TIGSINPUTFRAME_H_DEFINED
