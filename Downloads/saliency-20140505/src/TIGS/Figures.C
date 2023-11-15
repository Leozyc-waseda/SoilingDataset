/*!@file TIGS/Figures.C Generate illustration figures for TIGS programs */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/Figures.C $
// $Id: Figures.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef TIGS_FIGURES_C_DEFINED
#define TIGS_FIGURES_C_DEFINED

#include "TIGS/Figures.H"

#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/FilterOps.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/Range.H"
#include "Image/ShapeOps.H"
#include "TIGS/TigsInputFrame.H"
#include "TIGS/TrainingSet.H"
#include "Util/sformat.H"
#include "rutz/trace.h"

namespace
{
  Image<PixRGB<byte> > makeBargraph(const Image<double>& src,
                                    const float factor,
                                    const int height,
                                    int intercept,
                                    const int bar_width,
                                    const PixRGB<byte>& back,
                                    const PixRGB<byte>& fore)
  {
  GVX_TRACE(__PRETTY_FUNCTION__);
    Image<PixRGB<byte> > result(bar_width*src.getSize(), height, NO_INIT);
    result.clear(back);

    if (intercept < 0) intercept = 0;
    else if (intercept >= height) intercept = height-1;

    for (int i = 0; i < src.getSize(); ++i)
      {
        const int val = int(factor * src[i]);

        int pos = intercept-val;

        if (pos < 0) pos = 0;
        else if (pos >= height) pos = height-1;

        drawLine(result,
                 Point2D<int>(bar_width*i, pos),
                 Point2D<int>(bar_width*i, intercept),
                 fore, bar_width);
      }

    return result;
  }

  Image<PixRGB<byte> > addLabel(const Image<PixRGB<byte> >& img,
                                const PixRGB<byte>& color,
                                const char* label)
  {
    Image<PixRGB<byte> > img2(std::max(img.getWidth(), 144),
                              img.getHeight() + 30, NO_INIT);

    img2.clear(color);

    inplacePaste(img2, img, Point2D<int>(0,0));

    writeText(img2, Point2D<int>(0, img.getHeight()),
              label, PixRGB<byte>(0,0,0), color);

    return img2;
  }

  Image<PixRGB<byte> > stainImg(const Image<float>& img, const int zoom)
  {
    const Range<float> rr = rangeOf(img);

    const PixRGB<float> background(5.0f, 5.0f, 5.0f);
    const PixRGB<float> pos_stain(100.0f, 255.0f, 150.0f);
    const PixRGB<float> neg_stain(255.0f, 100.0f, 100.0f);

    return zoomXY(stainPosNeg(img,
                              std::max(fabs(rr.max()), fabs(rr.min())),
                              background, pos_stain, neg_stain),
                  zoom, zoom);
  }

  Point2D<int> findScaledMax(const Image<float>& img, const int zoom)
  {
    Point2D<int> loc;
    float maxval;
    findMax(img, loc, maxval);
    return Point2D<int>( int(zoom*(loc.i+0.5)), int(zoom*(loc.j+0.5)) );
  }
}

Image<PixRGB<byte> >
makeSumoDisplay(const TigsInputFrame& fin,
                const Image<float>& eyeposmap,
                const TrainingSet& tdata,
                const Point2D<int>& eyepos,
                const Image<float>& features)
{
  GVX_TRACE("makeSumoDisplay");

  if (fin.isGhost())
    LFATAL("makeSumoDisplay() needs non-ghost frames");

  const PixRGB<byte> col_eyepos1(255, 127, 0);
  const PixRGB<byte> col_eyepos2(160, 160, 160);
  const PixRGB<byte> col_histo_back(0, 15, 15);
  const PixRGB<byte> col_histo_fore(0, 255, 255);

  Image<PixRGB<byte> > drawframe(fin.origframe());

  drawPatch(drawframe, eyepos, 3, col_eyepos1);
  drawCircle(drawframe, eyepos, 30, col_eyepos1, 3);

  const int pos = tdata.p2p(eyepos);

  ASSERT(fin.lum().getDims() == fin.rg().getDims());
  ASSERT(fin.rg().getDims() == fin.by().getDims());

  const Image<PixRGB<byte> > imgs[4] =
    { rescale(drawframe, fin.lum().getDims()/2), decXY(fin.lum(), 2),
      decXY(fin.rg(), 2), decXY(fin.by(), 2) };

  const Image<PixRGB<byte> > arr = concatArray(imgs, 4, 2);

  Range<float> r = rangeOf(eyeposmap);

  static Range<float> rr;
  rr.merge(r);

  const PixRGB<float> background(15.0f, 15.0f, 15.0f);
  const PixRGB<float> pos_stain(15.0f, 255.0f, 100.0f);
  const PixRGB<float> neg_stain(255.0f, 15.0f, 50.0f);

  Image<PixRGB<byte> > bbiasmap
    (zoomXY(stainPosNeg(eyeposmap,
                        std::max(fabs(rr.max()), fabs(rr.min())),
                        background, pos_stain, neg_stain),
            tdata.inputReduction(),
            tdata.inputReduction()));

  drawPatch(bbiasmap, eyepos, 3, col_eyepos2);

  Image<PixRGB<byte> > bbiasmap2(std::max(bbiasmap.getWidth(), 144),
                                 bbiasmap.getHeight() + 50, ZEROS);

  inplacePaste(bbiasmap2, bbiasmap, Point2D<int>(0,0));

  writeText(bbiasmap2, Point2D<int>(0, bbiasmap.getHeight()),
            sformat("min: %g (%g)", r.min(), rr.min()).c_str(),
            PixRGB<byte>(neg_stain), PixRGB<byte>(0,0,0));

  writeText(bbiasmap2, Point2D<int>(0, bbiasmap.getHeight()+25),
            sformat("max: %g (%g)", r.max(), rr.max()).c_str(),
            PixRGB<byte>(pos_stain), PixRGB<byte>(0,0,0));

  const Image<PixRGB<byte> > fhisto =
    makeBargraph(features, 100.0f/255.0f, 100, 100, 1,
                 col_histo_back,
                 col_histo_fore);

  const Image<float> resiz_features =
    tdata.getFeatures().getHeight() > 1000
    ? decY(tdata.getFeatures(), 1 + tdata.getFeatures().getHeight() / 1000)
    : tdata.getFeatures();

  const Image<PixRGB<byte> > sumo_features = concatY(fhisto, Image<PixRGB<byte> >(resiz_features));

  const Image<PixRGB<byte> > eyehisto =
    makeBargraph(eyeposmap,
                 50.f/(std::max(fabs(r.min()),
                                fabs(r.max()))),
                 100, 50,
                 1,
                 col_histo_back,
                 col_histo_fore);

  const Image<float> resiz_positions =
    tdata.getPositions().getHeight() > 1000
    ? decY(tdata.getPositions(), 1 + tdata.getPositions().getHeight() / 1000)
    : tdata.getPositions();

  const Image<PixRGB<byte> > tp = resiz_positions * 255.0f;

  Image<PixRGB<byte> > sumo_eyepos = concatY(eyehisto, tp);

  drawLine(sumo_eyepos,
           Point2D<int>(pos, 0),
           Point2D<int>(pos, sumo_eyepos.getHeight()-1),
           col_eyepos1, 1);

  return concatLooseX(concatLooseY(arr, bbiasmap2),
                      concatX(sumo_features, sumo_eyepos));
}

Image<PixRGB<byte> >
makeSumoDisplay2(const TigsInputFrame& fin,
                 const Image<float>& tdmap,
                 const Image<float>& bumap,
                 const Image<float>& combomap,
                 const TrainingSet& tdata,
                 const Point2D<int>& eyepos)
{
  GVX_TRACE("makeSumoDisplay2");

  if (fin.isGhost())
    LFATAL("makeSumoDisplay2() needs non-ghost frames");

  const int zoom = tdata.inputReduction();

  const Point2D<int> tdpos = findScaledMax(tdmap, zoom);
  const Point2D<int> bupos = findScaledMax(bumap, zoom);
  const Point2D<int> combopos = findScaledMax(combomap, zoom);

  Image<PixRGB<byte> > drawframe(fin.origframe());

  const PixRGB<byte> col_human(255, 127, 0);
  const PixRGB<byte> col_td(255, 64, 64);
  const PixRGB<byte> col_bu(64, 64, 255);
  const PixRGB<byte> col_combo(192, 0, 255);

#define DRAWPOS(img)                            \
  drawPatch(img, tdpos, 3, col_td);             \
  drawCircle(img, tdpos, 18, col_td, 3);        \
  drawPatch(img, bupos, 3, col_bu);             \
  drawCircle(img, bupos, 18, col_bu, 3);        \
  drawPatch(img, combopos, 3, col_combo);       \
  drawCircle(img, combopos, 18, col_combo, 3);  \
  drawPatch(img, eyepos, 3, col_human);         \
  drawCircle(img, eyepos, 30, col_human, 3)

  DRAWPOS(drawframe);

  Image<PixRGB<byte> > btdmap = stainImg(tdmap, zoom);
  DRAWPOS(btdmap);

  Image<PixRGB<byte> > bbumap = stainImg(bumap, zoom);
  DRAWPOS(bbumap);

  Image<PixRGB<byte> > bcombomap = stainImg(combomap, zoom);
  DRAWPOS(bcombomap);

#undef DRAWPOS

  return concatLooseY
    (concatLooseX(addLabel(decXY(lowPass3(drawframe)), col_human, "input"),
                  addLabel(decXY(lowPass3(btdmap)), col_td, "top-down")),
     concatLooseX(addLabel(decXY(lowPass3(bbumap)), col_bu, "bottom-up"),
                  addLabel(decXY(lowPass3(bcombomap)), col_combo, "bu*td combo")));
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_FIGURES_C_DEFINED
