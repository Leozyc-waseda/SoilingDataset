/*!@file NeovisionII/Nv2LabelReader.H */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/Nv2LabelReader.H $
// $Id: Nv2LabelReader.H 14252 2010-11-19 17:43:55Z pez $
//

#ifndef NEOVISIONII_NV2LABELREADER_H_DEFINED
#define NEOVISIONII_NV2LABELREADER_H_DEFINED

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "NeovisionII/nv2_common.h"
#include "rutz/time.h"

#include <deque>
#include <string>

class Nv2LabelReader
{
public:

  Nv2LabelReader(const PixRGB<byte> color_,
                 const int label_reader_port,
                 const std::string& remote_patch_reader);

  ~Nv2LabelReader();

  void sendPatch(const uint32_t id,
                 const Image<PixRGB<byte> >& fullimg,
                 const Rectangle& foa,
                 const Image<PixRGB<byte> >& foapatch,
                 const rutz::time& qtime,
                 bool is_training_image,
                 const std::string& training_label,
                 const std::string& remote_command,
                 Point2D<int> fixLoc = Point2D<int>(0,0));

  struct LabeledImage
  {
    Image<PixRGB<byte> > img;
    std::string label;
    std::string ident;
  };

  LabeledImage getNextLabeledImage(bool ignore_nomatch,
                                   const size_t text_length,
                                   int FrameNumber = -1);

  double getLastConfidence() const;

private:
  Nv2LabelReader(const Nv2LabelReader&);
  Nv2LabelReader& operator=(const Nv2LabelReader&);

  struct PendingImage
  {
    Image<PixRGB<byte> > fullimg;
    Rectangle foa;
    uint32_t patch_id;
    rutz::time qtime;
  };

  struct nv2_label_reader* reader;
  const PixRGB<byte> color;
  enum nv2_pixel_type pixtype;
  std::deque<PendingImage> imgq;
  std::deque<rutz::time> times;
  rutz::atomic_int_t lastConfidence;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // NEOVISIONII_NV2LABELREADER_H_DEFINED
