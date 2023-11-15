/*!@file INVT/envisionpp.C simple program to exercise EnvVisualCortex */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/envisionpp.C $
// $Id: envisionpp.C 9841 2008-06-20 22:05:12Z lior $
//

#ifndef INVT_ENVISIONPP_C_DEFINED
#define INVT_ENVISIONPP_C_DEFINED

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Media/Streamer.H"
#include "Neuro/EnvVisualCortex.H"

class EnvisionStreamer : public Streamer
{
public:
  EnvisionStreamer()
    :
    Streamer("EnvisionStreamer"),
    itsEvc(new EnvVisualCortex(this->getManager()))
  {
    this->addComponent(itsEvc);
  }

private:
  virtual void onFrame(const GenericFrame& input,
                       FrameOstream& ofs,
                       const int frameNum)
  {
    itsEvc->input(input.asRgb());

    ofs.writeGray(itsEvc->getVCXmap(), "EVCO",
                  FrameInfo("copy of input frame", SRC_POS));
  }

  nub::ref<EnvVisualCortex> itsEvc;
};

int main(const int argc, const char **argv)
{
  EnvisionStreamer e;
  return e.run(argc, argv);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // INVT_ENVISIONPP_C_DEFINED
