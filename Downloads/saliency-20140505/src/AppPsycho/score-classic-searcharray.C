/*!@file AppPsycho/score-classic-searcharray.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/score-classic-searcharray.C $
// $Id: score-classic-searcharray.C 9490 2008-03-18 17:44:42Z rjpeters $
//

#ifndef APPPSYCHO_SCORE_CLASSIC_SEARCHARRAY_C_DEFINED
#define APPPSYCHO_SCORE_CLASSIC_SEARCHARRAY_C_DEFINED

#include "Image/CutPaste.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Range.H"
#include "Image/ShapeOps.H"
#include "Raster/GenericFrame.H"
#include "Raster/PfmParser.H"
#include "Raster/Raster.H"
#include "Util/StringConversions.H"
#include "Util/StringUtil.H"
#include "Util/log.H"

#include "GUI/XWinManaged.H"
#include "Image/Normalize.H"
#include "Image/DrawOps.H"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static int fixnum(const double x, const std::vector<double>& vals)
{
  int fixnum = 1;
  for (size_t i = 0; i < vals.size(); ++i)
    if (vals[i] >= x) ++fixnum;
  return fixnum;
}

int main(int argc, char** argv)
{
  if (argc != 6)
    {
      fprintf(stderr, "usage: %s salmap dims keyfile scorefile stimimage\n", argv[0]);
      return -1;
    }

  const std::string salmapfile = argv[1];
  const Dims rescaledims = fromStr<Dims>(argv[2]);
  const std::string keyfile = argv[3];
  const std::string scorefile = argv[4];
  const Image<byte> stimimage = Raster::ReadGray(argv[5]);

//   Image<PixRGB<byte> > stimmarkup(stimimage);

  const Image<float> salmap0 = PfmParser(salmapfile).getFrame().asFloat();
  const Image<float> salmap = rescaleBilinear(salmap0, rescaledims);

//   Image<PixRGB<byte> > salmarkup(normalizeFloat(salmap, FLOAT_NORM_0_255));;

  std::vector<double> bgmeanvals;
  std::vector<double> bgmaxvals;
  std::vector<double> fgmeanvals;
  std::vector<double> fgmaxvals;

  std::ifstream ifs(keyfile.c_str());
  if (!ifs.is_open())
    LFATAL("couldn't open '%s' for reading", keyfile.c_str());

  std::string line;
  while (std::getline(ifs, line))
    {
      if (line[0] == '%')
        continue;

      std::vector<std::string> toks;
      split(line, " ", std::back_inserter(toks));
      if (toks.size() != 5)
        LFATAL("malformed line: '%s'", line.c_str());

      const int x0 = fromStr<int>(toks[1]);
      const int y0 = fromStr<int>(toks[2]);
      const int x1 = fromStr<int>(toks[3]);
      const int y1 = fromStr<int>(toks[4]);

//       const Rectangle r = Rectangle::tlbrO(y0 + int(0.25*(y1-y0)),
//                                            x0 + int(0.25*(x1-x0)),
//                                            y0 + int(0.75*(y1-y0)),
//                                            x0 + int(0.75*(x1-x0)))
//         .getOverlap(salmap.getBounds());
      const Rectangle r = Rectangle::tlbrO(y0, x0, y1, x1)
        .getOverlap(salmap.getBounds());

      const Image<float> patch = crop(salmap, r, false);

      const float patchmean = mean(patch);
      const Range<float> patchrange = rangeOf(patch);

      if (fromStr<int>(toks[0]) == 0)
        {
          bgmeanvals.push_back(patchmean);
          bgmaxvals.push_back(patchrange.max());

//           drawRectSquareCorners(stimmarkup, r, PixRGB<byte>(0,0,255), 2);
//           drawRectSquareCorners(salmarkup, r, PixRGB<byte>(0,0,255), 2);
        }
      else if (fromStr<int>(toks[0]) == 1)
        {
          fgmeanvals.push_back(patchmean);
          fgmaxvals.push_back(patchrange.max());

//           drawRectSquareCorners(stimmarkup, r, PixRGB<byte>(255,0,0), 2);
//           drawRectSquareCorners(salmarkup, r, PixRGB<byte>(255,0,0), 2);
        }
      else
        LFATAL("invalid fg value '%s'", toks[0].c_str());
    }

  ifs.close();

  const Image<double> bgmean(&bgmeanvals[0], bgmeanvals.size(), 1);
  const Image<double> bgmax(&bgmaxvals[0], bgmaxvals.size(), 1);
  const Image<double> fgmean(&fgmeanvals[0], fgmeanvals.size(), 1);
  const Image<double> fgmax(&fgmaxvals[0], fgmaxvals.size(), 1);

  const double mean_bgmean = mean(bgmean);
  const double mean_bgmax = mean(bgmax);
  const double std_bgmean = stdev(bgmean);
  const double std_bgmax = stdev(bgmax);
  const double mean_fgmean = mean(fgmean);
  const double mean_fgmax = mean(fgmax);

  FILE* fwarn = fopen("warnings.txt", "a");
  if (fwarn == 0)
    LFATAL("couldn't open 'warnings.txt' for appending");
  bool bad = false;
#define CHECKNAN(x) if (isnan(x)) { bad = true; fprintf(fwarn, "nan: %s (args = %s %s %s %s)\n", #x, argv[1], argv[2], argv[3], argv[4]); }
  CHECKNAN(mean_bgmean);
  CHECKNAN(mean_bgmax);
  CHECKNAN(std_bgmean);
  CHECKNAN(std_bgmax);
  CHECKNAN(mean_fgmean);
  CHECKNAN(mean_fgmax);
  fclose(fwarn);

  if (bad) exit(0);

  double mean_nitems = 0.0;
  double mean_fgmean_zscore = 0.0;
  double mean_fgmax_zscore = 0.0;
  double mean_fgmean_over_mean_bgmean = 0.0;
  double mean_fgmax_over_mean_bgmax = 0.0;
  double mean_fgmean_minus_mean_bgmean = 0.0;
  double mean_fgmax_minus_mean_bgmax = 0.0;
  double mean_fgmean_fixnum = 0.0;
  double mean_fgmax_fixnum = 0.0;
  int n = 0;

  std::ifstream scorefs(scorefile.c_str());
  if (scorefs.is_open())
    {
      while (std::getline(scorefs, line))
        {
          std::vector<std::string> toks;
          split(line, " ", std::back_inserter(toks));
          if (toks.size() != 9)
            LFATAL("malformed line in score file: '%s'", line.c_str());

          ++n;
          mean_nitems += fromStr<double>(toks[0]);
          mean_fgmean_zscore += fromStr<double>(toks[1]);
          mean_fgmax_zscore += fromStr<double>(toks[2]);
          mean_fgmean_over_mean_bgmean += fromStr<double>(toks[3]);
          mean_fgmax_over_mean_bgmax += fromStr<double>(toks[4]);
          mean_fgmean_minus_mean_bgmean += fromStr<double>(toks[5]);
          mean_fgmax_minus_mean_bgmax += fromStr<double>(toks[6]);
          mean_fgmean_fixnum += fromStr<double>(toks[7]);
          mean_fgmax_fixnum += fromStr<double>(toks[8]);
        }
      scorefs.close();
    }

  FILE* f = fopen(scorefile.c_str(), "a");
  if (f == 0)
    LFATAL("couldn't open '%s' for appending", scorefile.c_str());
  // mean(fgmean_zscore) mean(fgmax_zscore)  mean(fgmean)/mean(bgmean) mean(fgmax)/mean(bgmax)  mean(fgmean)-mean(bgmean) mean(fgmax)-mean(bgmax)
  fprintf(f, "%12d %12g %12g %12g %12g %12g %12g %12d %12d\n",
          int(bgmeanvals.size()+fgmeanvals.size()),
          mean((fgmean - mean_bgmean) / std_bgmean),
          mean((fgmax - mean_bgmax) / std_bgmax),
          mean_fgmean / mean_bgmean,
          mean_fgmax / mean_bgmax,
          mean_fgmean - mean_bgmean,
          mean_fgmax - mean_bgmax,
          fixnum(mean_fgmean, bgmeanvals),
          fixnum(mean_fgmax, bgmaxvals));
  fprintf(stderr, "current: %12d %12g %12g %12g %12g %12g %12g %12d %12d\n",
          int(bgmeanvals.size()+fgmeanvals.size()),
          mean((fgmean - mean_bgmean) / std_bgmean),
          mean((fgmax - mean_bgmax) / std_bgmax),
          mean_fgmean / mean_bgmean,
          mean_fgmax / mean_bgmax,
          mean_fgmean - mean_bgmean,
          mean_fgmax - mean_bgmax,
          fixnum(mean_fgmean, bgmeanvals),
          fixnum(mean_fgmax, bgmaxvals));
  fclose(f);

  mean_nitems                    += int(bgmeanvals.size()+fgmeanvals.size());
  mean_fgmean_zscore             += mean((fgmean - mean_bgmean) / std_bgmean);
  mean_fgmax_zscore              += mean((fgmax - mean_bgmax) / std_bgmax);
  mean_fgmean_over_mean_bgmean   += mean_fgmean / mean_bgmean;
  mean_fgmax_over_mean_bgmax     += mean_fgmax / mean_bgmax;
  mean_fgmean_minus_mean_bgmean  += mean_fgmean - mean_bgmean;
  mean_fgmax_minus_mean_bgmax    += mean_fgmax - mean_bgmax;
  mean_fgmean_fixnum             += fixnum(mean_fgmean, bgmeanvals);
  mean_fgmax_fixnum              += fixnum(mean_fgmax, bgmaxvals);

  ++n;

  mean_nitems /= n;
  mean_fgmean_zscore /= n;
  mean_fgmax_zscore /= n;
  mean_fgmean_over_mean_bgmean /= n;
  mean_fgmax_over_mean_bgmax /= n;
  mean_fgmean_minus_mean_bgmean /= n;
  mean_fgmax_minus_mean_bgmax /= n;
  mean_fgmean_fixnum /= n;
  mean_fgmax_fixnum /= n;

  FILE* fsum = fopen(("sum-"+scorefile).c_str(), "w");
  if (fsum == 0)
    LFATAL("couldn't open 'sum-%s' for writing", scorefile.c_str());
  fprintf(fsum, "%-70s %3d %12g %12g %12g %12g %12g %12g %12g %12g %12g\n",
          ("sum-"+scorefile).c_str(),
          n,
          mean_nitems,
          mean_fgmean_zscore,
          mean_fgmax_zscore,
          mean_fgmean_over_mean_bgmean,
          mean_fgmax_over_mean_bgmax,
          mean_fgmean_minus_mean_bgmean,
          mean_fgmax_minus_mean_bgmax,
          mean_fgmean_fixnum,
          mean_fgmax_fixnum);
  fprintf(stderr, "overall: %3d %12g %12g %12g %12g %12g %12g %12g %12g %12g\n",
          n,
          mean_nitems,
          mean_fgmean_zscore,
          mean_fgmax_zscore,
          mean_fgmean_over_mean_bgmean,
          mean_fgmax_over_mean_bgmax,
          mean_fgmean_minus_mean_bgmean,
          mean_fgmax_minus_mean_bgmax,
          mean_fgmean_fixnum,
          mean_fgmax_fixnum);
  fclose(fsum);

//   XWinManaged xwin(concatX(stimmarkup, salmarkup));
//   while (!xwin.pressedCloseButton())
//     usleep(10000);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPPSYCHO_SCORE_CLASSIC_SEARCHARRAY_C_DEFINED
