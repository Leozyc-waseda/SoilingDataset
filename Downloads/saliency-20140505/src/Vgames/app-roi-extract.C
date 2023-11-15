/*!@file Vgames/app-roi-extract.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Vgames/app-roi-extract.C $
// $Id: app-roi-extract.C 15310 2012-06-01 02:29:24Z itti $
//

#ifndef VGAMES_APP_ROI_EXTRACT_C_DEFINED
#define VGAMES_APP_ROI_EXTRACT_C_DEFINED

#include "Component/ModelManager.H"
#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Layout.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Image/SimpleFont.H"
#include "Media/FrameSeries.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Raster/PfmParser.H"
#include "Raster/PfmWriter.H"
#include "Transport/FrameInfo.H"
#include "Util/Pause.H"
#include "Util/StringUtil.H"
#include "Util/csignals.H"
#include "Util/sformat.H"
#include "rutz/shared_ptr.h"

#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

#include <unistd.h>

Image<PixRGB<byte> > normC255(const Image<PixRGB<double> >& x)
{
  Image<PixRGB<double> > tmp(x);
  normalizeC(tmp, 0, 255);
  return Image<PixRGB<byte> >(tmp);
}

class PatchSet
{
public:
  PatchSet(const std::string& nm, const Dims& d)
    :
    itsName(nm),
    itsDims(d)
  {
    this->load();
  }

  Dims getDims() const { return itsDims; }

  void save() const
  {
    const std::string fname = sformat("%s.patchset", itsName.c_str());

    std::ofstream ofs(fname.c_str());
    if (!ofs.is_open())
      LFATAL("couldn't open %s for writing", fname.c_str());

    ofs << itsName << '\n';
    ofs << convertToString(itsDims) << '\n';

    std::map<std::string, PatchInfo>::const_iterator itr, stop;

    size_t ntotal = 0;

    for (itr = itsInfo.begin(), stop = itsInfo.end(); itr != stop; ++itr)
      {
        ofs << (*itr).first << '\n'
            << (*itr).second.id() << '\n'
            << (*itr).second.n() << '\n';

        ntotal += (*itr).second.n();
      }

    ASSERT(itsFeatures.size() == (itsDims.sz() * 3 * ntotal));

    const Image<float> features(&itsFeatures[0],
                                itsDims.sz() * 3, ntotal);

    Image<float> ids(itsInfo.size(), features.getHeight(), ZEROS);
    for (size_t i = 0; i < itsLabelIds.size(); ++i)
      {
        ASSERT(itsLabelIds[i] < ids.getWidth());
        ids.setVal(itsLabelIds[i], i, 1.0f);
      }

    PfmWriter::writeFloat(features,
                          sformat("%s.features.pfm", itsName.c_str()));

    PfmWriter::writeFloat(ids,
                          sformat("%s.ids.pfm", itsName.c_str()));
  }

  void load()
  {
    itsInfo.clear();

    const std::string fname = sformat("%s.patchset", itsName.c_str());

    std::ifstream ifs(fname.c_str());
    if (!ifs.is_open())
      {
        LINFO("couldn't open %s for reading", fname.c_str());
        return;
      }

    std::string name;
    std::getline(ifs, name);
    if (name != itsName)
      LFATAL("wrong name in file %s (expected %s, got %s)",
             fname.c_str(), itsName.c_str(), name.c_str());

    std::string dimsstr;
    std::getline(ifs, dimsstr);
    if (fromStr<Dims>(dimsstr) != itsDims)
      LFATAL("wrong dims in file %s (expected %s, got %s)",
             fname.c_str(), convertToString(itsDims).c_str(), dimsstr.c_str());

    size_t ntotal = 0;

    while (1)
      {
        std::string label;
        if (!std::getline(ifs, label))
          break;

        if (itsInfo.find(label) != itsInfo.end())
          LFATAL("already read PatchInfo for label %s in file %s",
                 label.c_str(), fname.c_str());

        int id;
        if (!(ifs >> id))
          LFATAL("couldn't read id value for PatchInfo %s from file %s",
                 label.c_str(), fname.c_str());
        ifs >> std::ws;
        theirNextLabelId = std::max(theirNextLabelId, id + 1);
        LINFO("got patch id = %d, next id = %d",
              id, theirNextLabelId);

        int n = -1;
        if (!(ifs >> n))
          LFATAL("couldn't read N value for PatchInfo %s from file %s",
                 label.c_str(), fname.c_str());
        ifs >> std::ws;
        if (n < 0)
          LFATAL("got bogus N value %d for PatchInfo %s from file %s",
                 n, label.c_str(), fname.c_str());

        ntotal += n;

        PatchInfo info(label, id, n);

        itsInfo.insert(std::make_pair(label, info));

        LINFO("read PatchInfo %s with n=%" ZU " from file %s",
              label.c_str(), info.n(), fname.c_str());
      }

    const Image<float> features =
      PfmParser(sformat("%s.features.pfm", itsName.c_str())).getFrame().asFloat();

    ASSERT(features.getHeight() == int(ntotal));
    ASSERT(features.getWidth() == itsDims.sz() * 3);

    const Image<float> ids =
      PfmParser(sformat("%s.ids.pfm", itsName.c_str())).getFrame().asFloat();

    ASSERT(ids.getHeight() == int(ntotal));
    LINFO("ids.getWidth() = %d", ids.getWidth());
    LINFO("theirNextLabelId = %d", theirNextLabelId);
    ASSERT(ids.getWidth() == theirNextLabelId);

    itsFeatures.resize(0);
    itsFeatures.insert(itsFeatures.end(), features.begin(), features.end());

    itsLabelIds.resize(0);
    for (int y = 0; y < ids.getHeight(); ++y)
      {
        int pos = -1;
        for (int x = 0; x < ids.getWidth(); ++x)
          {
            const float val = ids.getVal(x,y);
            if (val == 1.0f)
              {
                if (pos == -1)
                  pos = x;
                else
                  LFATAL("oops! more than one label id (columns %d and %d) "
                         "in row %d of file %s.ids.pfm",
                         pos, x, y, itsName.c_str());
              }
            else if (val != 0.0f)
              {
                LFATAL("oops! invalid value %.17f in column %d, row %d "
                       "of file %s.ids.pfm", val, x, y, itsName.c_str());
              }
          }
        if (pos == -1)
          LFATAL("oops! no label id in row %d of file %s.ids.pfm",
                 y, itsName.c_str());

        ASSERT(pos >= 0 && pos < theirNextLabelId);

        itsLabelIds.push_back(pos);
      }
  }

  void addLabeledPatch(const std::string& label,
                       const Image<PixRGB<byte> >& patch)
  {
    ASSERT(patch.getDims() == itsDims);

    if (itsInfo.find(label) == itsInfo.end())
      itsInfo.insert(std::make_pair(label, PatchInfo(label, theirNextLabelId++)));

    PatchInfo& info = (*itsInfo.find(label)).second;

    info.addPatch(patch);

    for (int i = 0; i < patch.getSize(); ++i)
      for (int j = 0; j < 3; ++j)
        itsFeatures.push_back(float(patch.getVal(i).p[j]));

    itsLabelIds.push_back(info.id());
  }

private:
  const std::string itsName;
  const Dims itsDims;

  struct PatchInfo
  {
    PatchInfo(const std::string& l, int id, size_t n = 0)
      : itsLabel(l), itsLabelId(id), itsN(n) {}

    int id() const { return itsLabelId; }

    size_t n() const { return itsN; }

    void addPatch(const Image<PixRGB<byte> >& patch)
    {
      itsN++;
    }

  private:
    const std::string itsLabel;
    const int itsLabelId;
    size_t itsN;
  };

  static int theirNextLabelId;

  std::map<std::string, PatchInfo> itsInfo;
  std::vector<float> itsFeatures;
  std::vector<int> itsLabelIds;
};

int PatchSet::theirNextLabelId = 0;

class RoiExtractor
{
public:
  RoiExtractor(const std::string& nm,
               const rutz::shared_ptr<PatchSet>& ps,
               const Point2D<int>& pt)
    :
    itsName(nm),
    itsPatchSet(ps),
    itsRegion(Rectangle(pt, ps->getDims()))
  {}

  void label(Image<PixRGB<byte> >& img, const PixRGB<byte>& col)
  {
    drawRectSquareCorners(img, itsRegion, col, 1);

    writeText(img, itsRegion.bottomLeft(), itsName.c_str(),
              col, PixRGB<byte>(0,0,0),
              SimpleFont::FIXED(6), true);
  }

  const Rectangle& rect() const { return itsRegion; }

  PatchSet& patchSet() { return *itsPatchSet; }

private:
  const std::string itsName;
  const rutz::shared_ptr<PatchSet> itsPatchSet;
  const Rectangle itsRegion;
};

int main(const int argc, const char **argv)
{
  volatile int signum = 0;
  catchsignals(&signum);

  ModelManager manager("Streamer");

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  if (manager.parseCommandLine(argc, argv, "configfile", 1, 2) == false)
    return(1);

  std::map<std::string, rutz::shared_ptr<PatchSet> > patches;
  std::vector<rutz::shared_ptr<RoiExtractor> > regions;

  {
    std::ifstream ifs(manager.getExtraArg(0).c_str());
    if (!ifs.is_open())
      LFATAL("couldn't open %s for reading", manager.getExtraArg(0).c_str());

    std::string line;
    while (std::getline(ifs, line))
      {
        if (line.length() > 0 && line[0] == '#')
          continue;

        std::vector<std::string> parts;
        split(line, ":", std::back_inserter(parts));

        if (parts.size() == 0)
          LFATAL("invalid empty argument");

        if (parts[0] == "patchset")
          {
            if (parts.size() != 3)
              LFATAL("expected patchset:name:dims but got %s",
                     line.c_str());

            const std::string nm = parts[1];
            const Dims d = fromStr<Dims>(parts[2]);
            patches[nm] = rutz::shared_ptr<PatchSet>(new PatchSet(nm, d));
          }
        else if (parts[0] == "roi")
          {
            if (parts.size() != 4)
              LFATAL("expected roi:name:patchsetname:point but got %s",
                     line.c_str());

            rutz::shared_ptr<PatchSet> p = patches[parts[2]];
            if (p.get() == 0)
              LFATAL("invalid patchset name %s", parts[1].c_str());

            const Point2D<int> pt = fromStr<Point2D<int> >(parts[3]);

            regions.push_back(rutz::shared_ptr<RoiExtractor>
                              (new RoiExtractor(parts[1], p, pt)));
          }
      }
  }

  std::string outprefix = "regions";

  manager.start();

  ifs->startStream();

  PauseWaiter p;

//   XWinManaged mainwin(ifs->peekDims(), -1, -1, "main");
  XWinManaged zoomwin(Dims(16,16), -1, -1, "zoom");

  std::ifstream autoresp;
  if (manager.numExtraArgs() >= 2)
    {
      autoresp.open(manager.getExtraArg(1).c_str());
      if (!autoresp.is_open())
        LFATAL("couldn't open %s for reading",
               manager.getExtraArg(1).c_str());
    }

  int n = 0;

  while (true)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          return -1;
        }

      if (p.checkPause())
        continue;

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      const Image<PixRGB<byte> > input = ifs->readRGB();
      if (!input.initialized())
        break;

      Image<PixRGB<byte> > labeledinput(input);
      for (size_t i = 0; i < regions.size(); ++i)
        regions[i]->label(labeledinput, PixRGB<byte>(255, 0, 0));

//       mainwin.setDims(input.getDims());

      bool doquit = false;

      for (size_t i = 0; i < regions.size(); ++i)
        {
          Image<PixRGB<byte> > inputcopy(labeledinput);
          regions[i]->label(inputcopy, PixRGB<byte>(128, 255, 0));
//           mainwin.drawImage(inputcopy);

          const Image<PixRGB<byte> > patch = crop(input, regions[i]->rect());
          Image<PixRGB<byte> > zoomed = zoomXY(patch, 8);
          zoomwin.setDims(zoomed.getDims());

          std::string resp;

          if (autoresp.is_open())
            {
              std::string line;
              if (!std::getline(autoresp, line))
                {
                  LERROR("couldn't read line %d of autoresponse file", n);
                  break;
                }

              std::istringstream iss(line);
              int nn;
              if (!(iss >> nn >> resp))
                LFATAL("couldn't parse number and response from "
                       "line %d of autoresponse file", n);

              if (n != nn)
                LFATAL("wrong frame number in autoresponse file "
                       "(got %d, expected %d)", nn, n);

              writeText(zoomed, Point2D<int>(0,0), line.c_str(),
                        PixRGB<byte>(0,0,255), PixRGB<byte>(0,0,0),
                        SimpleFont::FIXED(10), true);
              zoomwin.drawImage(zoomed);
           }
          else
            {
              zoomwin.drawImage(zoomed);

              while ((resp = zoomwin.getLastKeyString()).length() == 0)
                {
                  usleep(10000);
                }

              if (isalnum(resp[0]))
                resp = resp[0];
              else if (resp[0] == ' ') // space
                resp = "none";
              else if (resp[0] == '?')
                resp = "unknown";
              else if (resp[0] == 27) // escape
                {
                  LINFO("ESCAPE!");
                  doquit = true;
                  break;
                }
              else // invalid response
                {
                  resp = "";
                }
            }

          if (resp.length() > 0 && resp != "unknown")
            regions[i]->patchSet().addLabeledPatch(resp, patch);
        }

      if (doquit)
        break;

      ++n;
    }

  for (std::map<std::string, rutz::shared_ptr<PatchSet> >::const_iterator
         itr = patches.begin(), stop = patches.end(); itr != stop; ++itr)
    {
      (*itr).second->save();
    }

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // VGAMES_APP_ROI_EXTRACT_C_DEFINED
