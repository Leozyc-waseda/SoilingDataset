/*!@file NeovisionII/Nv2LabelReader.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/Nv2LabelReader.C $
// $Id: Nv2LabelReader.C 14252 2010-11-19 17:43:55Z pez $
//

#ifndef NEOVISIONII_NV2LABELREADER_C_DEFINED
#define NEOVISIONII_NV2LABELREADER_C_DEFINED


#include <deque>
#include <string>

#include "NeovisionII/Nv2LabelReader.H"

#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "NeovisionII/nv2_label_reader.h"
#include "Util/StringConversions.H"
#include "Util/StringUtil.H"
#include "Util/sformat.H"

// ######################################################################
template <class T>
void writeText2(Image<T>& dst,
               const Point2D<int>& pt, const char* text,
               const T col, const T bgcol, const SimpleFont& f,
               const double bg_alpha,
               const TextAnchor anchor)
{
  const int textwidth = strlen(text) * f.w();
  const int textheight = f.h();

  const Point2D<int> top_left
    =    anchor == ANCHOR_BOTTOM_RIGHT ?    pt - Point2D<int>(textwidth, textheight)
    :    anchor == ANCHOR_BOTTOM_LEFT  ?    pt - Point2D<int>(0, textheight)
    :    anchor == ANCHOR_TOP_RIGHT    ?    pt - Point2D<int>(textwidth, 0)
    : /* anchor == ANCHOR_TOP_LEFT     ? */ pt;

  Point2D<int> p = top_left; // copy for modif
  const int ww = dst.getWidth(), hh = dst.getHeight();
  const int len = int(strlen(text));

  for (int i = 0; i < len; i ++)
    {
      const unsigned char *ptr = f.charptr(text[i]);

      for (int y = 0; y < int(f.h()); y ++)
        for (int x = 0; x < int(f.w()); x ++)
          if (p.i + x >= 0 && p.i + x < ww && p.j + y >= 0 && p.j + y < hh)
            {
              if (!ptr[y * f.w() + x])
                dst.setVal(p.i + x, p.j + y, col);
              else
                dst.setVal(p.i + x, p.j + y,
                           bgcol * (1.0 - bg_alpha)
                           + dst.getVal(p.i + x, p.j + y) * bg_alpha);
            }
      p.i += f.w();
    }
}

// ######################################################################
Nv2LabelReader::Nv2LabelReader(const PixRGB<byte> color_,
                               const int label_reader_port,
                               const std::string& remote_patch_reader)
  :
  reader(),
  color(color_),
  lastConfidence()
{
  std::vector<std::string> parts;
  split(remote_patch_reader, ":", std::back_inserter(parts));
  if (parts.size() != 2 && parts.size() != 3)
    LFATAL("couldn't parse addr:port[:pixtype] from '%s'",
           remote_patch_reader.c_str());

  const std::string remote_patch_reader_addr = parts[0];
  const int remote_patch_reader_port = fromStr<int>(parts[1]);

  //this->pixtype = NV2_PIXEL_TYPE_GRAY8;
  this->pixtype = NV2_PIXEL_TYPE_RGB24;
  if (parts.size() >= 3)
    {
      if (parts[2].compare("gray8") == 0)
        this->pixtype = NV2_PIXEL_TYPE_GRAY8;
      else if (parts[2].compare("rgb24") == 0)
        this->pixtype = NV2_PIXEL_TYPE_RGB24;
      else
        LFATAL("invalid pixel type %s (expected gray8 or rgb24",
               parts[2].c_str());
    }

  reader = nv2_label_reader_create(label_reader_port,
                                   remote_patch_reader_addr.c_str(),
                                   remote_patch_reader_port);

  LINFO("label reader at %s:%d, "
        "listening for labels on port %d",
        remote_patch_reader_addr.c_str(),
        remote_patch_reader_port,
        label_reader_port);
}

// ######################################################################
Nv2LabelReader::~Nv2LabelReader()
{
  nv2_label_reader_destroy(reader);
}

// ######################################################################
void Nv2LabelReader::sendPatch(const uint32_t id,
                               const Image<PixRGB<byte> >& fullimg,
                               const Rectangle& foa,
                               const Image<PixRGB<byte> >& foapatch,
                               const rutz::time& qtime,
                               bool is_training_image,
                               const std::string& training_label,
                               const std::string& remote_command,
                               Point2D<int> fixLoc)
{
  {
    const size_t npix = foapatch.getSize();

    nv2_image_patch patch;
    patch.protocol_version = NV2_PATCH_PROTOCOL_VERSION;
    patch.width = foapatch.getWidth();
    patch.height = foapatch.getHeight();
    patch.fix_x = fixLoc.i;
    patch.fix_y = fixLoc.j;
    patch.id = id;
    patch.is_training_image = is_training_image ? 1 : 0;
    patch.type = this->pixtype;
    nv2_image_patch_set_training_label(&patch, training_label.c_str());
    nv2_image_patch_set_remote_command(&patch, remote_command.c_str());

    switch (this->pixtype)
      {
      case NV2_PIXEL_TYPE_NONE:
        patch.data = 0;
        break;

      case NV2_PIXEL_TYPE_GRAY8:
        {
          patch.data = (unsigned char*) malloc(npix * sizeof(byte));
          if (patch.data == 0)
            LFATAL("malloc() failed");

          const Image<PixRGB<byte> >::const_iterator foaptr =
            foapatch.begin();

          for (size_t i = 0; i < npix; ++i)
            patch.data[i] = foaptr[i].luminance();
        }
        break;

      case NV2_PIXEL_TYPE_RGB24:
        {
          patch.data = (unsigned char*) malloc(3 * npix * sizeof(byte));
          if (patch.data == 0)
            LFATAL("malloc() failed");

          memcpy(&patch.data[0], foapatch.getArrayPtr(),
                 3 * npix * sizeof(byte));
        }
        break;
      }

    nv2_label_reader_send_patch(reader, &patch);
  }

  PendingImage qimg;
  qimg.fullimg = fullimg;
  qimg.foa = foa;
  qimg.patch_id = id;
  qimg.qtime = rutz::time::wall_clock_now();

  imgq.push_back(qimg);

  // if the queue gets too large, just drop some old frames so that
  // we don't allow unbounded memory usage
  while (imgq.size() > 60)
    {
      imgq.pop_front();
    }
}

// ######################################################################
// Added by PEZ, not fully tested yet
// ######################################################################

class FilterLabel
{
private:
  float itsMaxCenterDist2;
  int itsForgetIfMissingFor;
  int itsFilterLength;
  
  struct CacheItem
  {
    CacheItem(const Point2D<int>& center, const std::string& label, int frameNumber) :
      itsCenter(center),
      itsLabel(label),
      itsFrameNumber(frameNumber)
    {
    }
    
    CacheItem(const CacheItem& that) :
      itsCenter(that.itsCenter),
      itsLabel(that.itsLabel),
      itsFrameNumber(that.itsFrameNumber)
    {
    }
    
    CacheItem& operator=(const CacheItem& that)
    {
      itsCenter = that.itsCenter;
      itsLabel = that.itsLabel;
      itsFrameNumber = that.itsFrameNumber;
      
      return *this;
    }
    
    float dist2(const Point2D<int>& center)
    {
      float dx = center.i - itsCenter.i;
      float dy = center.j - itsCenter.j;
      return dx * dx + dy * dy;
    }
    
    float dist2(const Rectangle& rect)
    {
      return dist2(rect.center());
    }

    Point2D<int> itsCenter;
    std::string itsLabel;
    int itsFrameNumber;
  };
  
  typedef std::deque<CacheItem> cacheSingleItem;
  std::list<cacheSingleItem> cache;

public:
  FilterLabel(float maxCenterDist = 64.0f, int forgetIfMissingFor = 10, int filterLength = 9) :
    itsMaxCenterDist2(maxCenterDist * maxCenterDist),
    itsForgetIfMissingFor(forgetIfMissingFor),
    itsFilterLength(filterLength)
  {
  }
  
  virtual ~FilterLabel()
  {
  }
  
  std::string findLabel(cacheSingleItem& singleItem)
  {
    std::map<std::string,int> count;
    
    std::deque<CacheItem>::iterator itr = singleItem.begin();
    const std::deque<CacheItem>::iterator end = singleItem.end();
    
    for (; itr != end; ++itr)
    {
      ++count[itr->itsLabel];
    }
    
    std::map<std::string,int>::iterator itr2 = count.begin();
    std::map<std::string,int>::iterator end2 = count.end();
    
    std::string label;
    int max = 0;
    for (; itr2 != end2; ++itr2)
    {
      if (max == 0 || max < (*itr2).second)
      {
        max = (*itr2).second;
        label =(*itr2).first;
      }
    }
    
    return label;
  }
  
  std::string push(const Rectangle& rect, const std::string& label, int frameNumber)
  {
    Point2D<int> center = rect.center();
    
    std::list<cacheSingleItem>::iterator itr = cache.begin();
    const std::list<cacheSingleItem>::iterator end = cache.end();
    std::list<cacheSingleItem>::iterator best = end;
    float bestDist2 = itsMaxCenterDist2 * 2;
    
    for (; itr != end; ++itr)
    {
      cacheSingleItem& singleItem = *itr;
      float d2 = singleItem.back().dist2(center);
      if (best == end || d2 < bestDist2)
      {
        bestDist2 = d2;
        best = itr;
      }
    }
    
    if (best == end || bestDist2 > itsMaxCenterDist2)
    {  // add this new item
      cacheSingleItem singleItem;
      singleItem.push_back(CacheItem(center, label, frameNumber));
      cache.push_back(singleItem);
      return label;
    }
    
    if (best->size() >= static_cast<unsigned int>(itsFilterLength))
      best->pop_front(); // remove the old data from cache
      
    best->push_back(CacheItem(center, label, frameNumber));
    
    return findLabel(*best);
  }
  
  void prune(int currentFrame)
  {
    std::list<cacheSingleItem>::iterator itr = cache.begin();
    const std::list<cacheSingleItem>::iterator end = cache.end();
    
    while (itr != end)
    {
      cacheSingleItem& singleItem = *itr;
      if (singleItem.back().itsFrameNumber < currentFrame - itsForgetIfMissingFor)
      { // too old, let's forget it
        itr = cache.erase(itr);
      }
      else
      {
        ++itr;
      }
    }
  }
};

FilterLabel filterLabel;

// ######################################################################
Nv2LabelReader::LabeledImage
Nv2LabelReader::getNextLabeledImage(bool ignore_nomatch,
                                    const size_t text_length,
                                    int FrameNumber)
{
  LabeledImage result;

  if (imgq.size() == 0)
    return result; // with a still-empty image

  struct nv2_patch_label label;
  const int gotit =
    nv2_label_reader_get_current_label(reader, &label);

  if (!gotit)
    return result; // with a still-empty image

  // else ...

  lastConfidence.atomic_set(label.confidence);

  result.ident = label.source;
  result.label = label.name;
  

  while (imgq.size() > 0 && imgq.front().patch_id < label.patch_id)
    // forget about patches that have been skipped by the label
    // server:
    imgq.pop_front();

  if (imgq.size() == 0 || imgq.front().patch_id > label.patch_id)
    return result; // with a still-empty image

  ASSERT(imgq.size() > 0 && imgq.front().patch_id == label.patch_id);

  PendingImage qimg = imgq.front();
  imgq.pop_front();

  if (ignore_nomatch &&
      (strncmp(label.name, "nomatch", 7) == 0 ||
       strncmp(label.name, "none", 4) == 0))
    return result; // with a still-empty image

  LINFO("label.name = '%s'", label.name);

  const rutz::time now = rutz::time::wall_clock_now();

  times.push_back(now);

  const double fps =
    times.size() >= 2
    ? (times.size() - 1) / (times.back() - times.front()).sec()
    : 0.0;

  if (times.size() > 2 && (times.back() - times.front()).sec() > 3.0)
    times.pop_front();

  if (!qimg.foa.isValid())
    return result;

  // added filter by PEZ
  if (FrameNumber >= 0)
  {
    std::string tempLabel = filterLabel.push(qimg.foa, result.label, FrameNumber);
    if (result.label != tempLabel)
    {
      LINFO("Label '%s' is relaced by '%s'", result.label.c_str(), tempLabel.c_str());
      result.label = tempLabel;
      strncpy(label.name, tempLabel.c_str(), sizeof(label.name));
      label.name[sizeof(label.name)-1]=0;
      label.confidence = 0.0;
    }
    filterLabel.prune(FrameNumber);
  }

  drawRectSquareCorners(qimg.fullimg, qimg.foa, this->color, 3);

  Point2D<int> textpos((qimg.foa.left() + qimg.foa.rightO()) / 2,
                  (qimg.foa.top() + qimg.foa.bottomO()) / 2);

  TextAnchor textloc = ANCHOR_TOP_LEFT;

  if (textpos.i < qimg.fullimg.getWidth() / 2)
    {
      if (textpos.j < qimg.fullimg.getHeight() / 2)
        {
          textloc = ANCHOR_TOP_LEFT;
          textpos.i += 10;
          textpos.j += 10;
        }
      else
        {
          textloc = ANCHOR_BOTTOM_LEFT;
          textpos.i += 10;
          textpos.j -= 10;
        }
    }
  else
    {
      if (textpos.j < qimg.fullimg.getHeight() / 2)
        {
          textloc = ANCHOR_TOP_RIGHT;
          textpos.i -= 10;
          textpos.j += 10;
        }
      else
        {
          textloc = ANCHOR_BOTTOM_RIGHT;
          textpos.i -= 10;
          textpos.j -= 10;
        }
    }

  writeText2(qimg.fullimg,
            textpos,
            sformat("%s (%.2f)", label.name, double(label.confidence) / double(NV2_MAX_LABEL_CONFIDENCE)).c_str(),
            this->color, PixRGB<byte>(0,0,0),
            SimpleFont::FIXED(14),
            0.5,
            textloc);

  const std::string lines[3] =
    {
      sformat("[c=%4.2f] %s",
              double(label.confidence) / double(NV2_MAX_LABEL_CONFIDENCE),
              label.name),
      sformat("%s", label.extra_info),
      sformat("%s: lag %06.3fs #%06u [%5.2ffps]",
              label.source,
              (now - qimg.qtime).sec(),
              (unsigned int) label.patch_id,
              fps)
    };

  const Image<PixRGB<byte> > textarea =
    makeMultilineTextBox(qimg.fullimg.getWidth(), &lines[0], 3,
                         this->color, PixRGB<byte>(0,0,0),
                         text_length);

  result.img = concatY(qimg.fullimg, textarea);
  return result;
}

// ######################################################################
double Nv2LabelReader::getLastConfidence() const
{
  return (double(lastConfidence.atomic_get())
          / double(NV2_MAX_LABEL_CONFIDENCE));
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // NEOVISIONII_NV2LABELREADER_C_DEFINED
