/*!@file INVT/simple-saliency.C A very simple saliency map implementation */

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
// Primary maintainer for this file: Laurent Itti <itti@pollux.usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/simple-saliency.C $
// $Id: simple-saliency.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Util/log.H"          // logging facilities, provides LINFO(), LFATAL(), etc functions
#include "Util/sformat.H"      // sformat() is similar to sprintf() but returns a C++ string

#include "Image/PixelsTypes.H" // for PixRGB<T>, etc
#include "Image/Image.H"       // template image class
#include "Image/ImageSet.H"    // for ImageSet<T>, a set of images (e.g., a pyramid)

#include "Image/ColorOps.H"    // for luminance(), getRGBY(), etc
#include "Image/PyramidOps.H"  // for buildPyrGeneric(), centerSurround(), etc
#include "Image/PyrBuilder.H"  // for the various builders for different types of pyramids
#include "Image/ShapeOps.H"    // for downSize()
#include "Image/fancynorm.H"   // for maxNormalize()
#include "Image/ImageSetOps.H" // for doLowThresh()

#include "Raster/Raster.H"     // for reading/writing images from/to disk
#include "Image/DrawOps.H"     // for colGreyCombo(), used only to create an output display
#include "Image/MathOps.H"     // for getMinMax, used only to print informational messages
#include "Image/Normalize.H"   // for normalizeFloat(), used only to write outputs to disk


// This program is a simplified version of the Itti et al (1998) visual saliency algorithm, designed for didactic
// purposes. Here, we compute a saliency map as if you were using: ezvision --maxnorm-type=Maxnorm --nouse-random

// To understand what is happening in this program, you should:
//
// - familiarize yourself with the Dims class in src/Image/Dims.H; it's just a simple struct with a width and a height
//   of an image;
//
// - familiarize yourself with the PixRGB<T> pixel type in src/Image/PixelsTypes.H; it's just a simple struct with 3
//   components for red, green and blue;
//
// - familiarize yourself with the Image<T> class in src/Image.H; it's just a 2D array of pixels with some Dims; you do
//   not need to worry about the internal handling of the array (copy-on-write and ref-counting). But check out the
//   begin() and beginw() functions as those are how one gets direct access to the raw pixel array through C++
//   iterators (linear access, scanning the image like a TV gun);
//
// - familiarize yourself with ImageSet<T>, it's just a 1D array of images, we use it here to store image pyramids;
//
// - then, have a look at the following functions used in the present program:
//
//   getRGBY() in Image/ColorOps.C
//   downSize() in Image/ShapeOps.C
//   centerSurround() in Image/PyramidOps.C, which internally uses centerSurround() of Image/FilterOps.C; note that this
//     one in turns internally uses inplaceAttenuateBorders() defined in Image/MathOps.C
//   maxNormalize in Image/fancynorm.C, here we use the VCXNORM_MAXNORM version of this algo
//   buildPyrGeneric() in Image/PyramidOps.C -- this is probably the most complicated one, it internally calls
//     builders for various types of image pyramids, which are defined in Image/PyrBuilder.C
//   doLowThresh() in ImageSetOps.C, user by the motion channel

// The following papers will help:
//
// * http://ilab.usc.edu/publications/doc/Itti_etal98pami.pdf
//   Check out Fig 1 for the general flow diagram, and Fig. 2 for the max-normalization operator. Note that eq. 2 and 3
//   in this paper have a typo.
//
// * http://ilab.usc.edu/publications/doc/Itti_Koch00vr.pdf
//   Check out Fig. 2.a for how the center-surround operations work. Note that this also introduces a different
//   max-normalization scheme (Fog. 2b) which is not used in the present program.
//
// * http://ilab.usc.edu/publications/doc/Itti_Koch01ei.pdf
//   Have a look at Fig. 4 for the "truncated filter" boundary condition which we use to create image pyramids.
//
// * http://ilab.usc.edu/publications/doc/Itti_etal03spienn.pdf
//   This version of the model includes flicker and motion channels and all the equations are correct, so use this as
//   the reference for the full algorithm.

// ######################################################################
// ##### Global options:
// ######################################################################
#define sml            4   /* pyramid level of the saliency map */
#define level_min      2   /* min center level */
#define level_max      4   /* max center level */
#define delta_min      3   /* min center-surround delta */
#define delta_max      4   /* max center-surround delta */
#define maxdepth       (level_max + delta_max + 1)
#define num_feat_maps  ((level_max - level_min + 1) * (delta_max - delta_min + 1)) /* num of feture maps per pyramid */
#define motionlowthresh 3.0F

// ######################################################################
//! Compute the luminance of an RGB image, and convert from byte to float pixels
/*! note that we could as well use luminance() defined in Image/ColorOps.H except that the rounding side effects are
    slightly different. This function corresponds to the one we use in our master code (it's burried in
    Channels/InputFrame.C).*/
Image<float> computeLuminance(const Image<PixRGB<byte> >& img)
{
  Image<float> lum(img.getDims(), NO_INIT);
  Image<PixRGB<byte> >::const_iterator in = img.begin(), stop = img.end();
  Image<float>::iterator dest = lum.beginw();
  const float one_third = 1.0F / 3.0F;

  while(in != stop) {
    *dest++ = one_third * (in->red() + in->green() + in->blue());
    ++in;
  }

  return lum;
}

// ######################################################################
//! Max-normalize a feature map or conspicuity map
Image<float> normalizMAP(const Image<float>& ima, const Dims& dims, const char *label)
{
  // do we want to downsize the map?
  Image<float> dsima;
  if (dims.isNonEmpty()) dsima = downSize(ima, dims); else dsima = ima;

  // apply spatial competition for salience (mx-normalization):
  Image<float> result = maxNormalize(dsima, MAXNORMMIN, MAXNORMMAX, VCXNORM_MAXNORM);

  // show some info about this map:
  float mi, ma, nmi, nma; getMinMax(ima, mi, ma); getMinMax(result, nmi, nma);
  LINFO("%s: raw range [%f .. %f] max-normalized to [%f .. %f]", label, mi, ma, nmi, nma);

  return result;
}

// ######################################################################
// Compute a conspicuity map, called with various pyramid types for the various channels
Image<float> computeCMAP(const Image<float>& fima, const PyramidType ptyp, const char *label,
                         const bool absol, const float ori = 0.0F)
{
  LINFO("Building %s channel:", label);

  // compute pyramid:
  ImageSet<float> pyr = buildPyrGeneric(fima, 0, maxdepth, ptyp, ori);

  // alloc conspicuity map and clear it:
  Image<float> cmap(pyr[sml].getDims(), ZEROS);

  // get all the center-surround maps and combine them:
  for (int delta = delta_min; delta <= delta_max; ++delta)
    for (int lev = level_min; lev <= level_max; ++lev)
      {
        Image<float> tmp = centerSurround(pyr, lev, lev + delta, absol);
        std::string lbl = sformat("  %s(%d,%d)", label, lev, lev + delta);

        tmp = normalizMAP(tmp, cmap.getDims(), lbl.c_str());

        cmap += tmp;
      }

  float mi, ma;
  getMinMax(cmap, mi, ma); LINFO("%s: final cmap range [%f .. %f]", label, mi, ma);

  return cmap;
}

// ######################################################################
// Compute a motion conspicuity map from a motion pyramid
Image<float> computeMMAP(ImageSet<float>& pyr, const char *label)
{
  LINFO("Building %s channel:", label);

  // apply a low threshold to cut small motion values:
  doLowThresh(pyr, motionlowthresh);

  // alloc conspicuity map and clear it:
  Image<float> cmap(pyr[sml].getDims(), ZEROS);

  // get all the center-surround maps and combine them:
  for (int delta = delta_min; delta <= delta_max; ++delta)
    for (int lev = level_min; lev <= level_max; ++lev)
      {
        Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
        std::string lbl = sformat("  %s(%d,%d)", label, lev, lev + delta);

        tmp = normalizMAP(tmp, cmap.getDims(), lbl.c_str());

        cmap += tmp;
      }

  float mi, ma;
  getMinMax(cmap, mi, ma); LINFO("%s: final cmap range [%f .. %f]", label, mi, ma);

  return cmap;
}

// ######################################################################
//! Simple saliency map computation
/*! This is a barebones implementation of the Itti, Koch & Niebur (1998) saliency map algorithm */
int main(const int argc, const char **argv)
{
  if (argc != 4) LFATAL("Usage: %s <input-stem> <traj-stem> <sm-stem>\n"
                        "Will read <input-stem>000000.png, <input-stem>000001.png, etc and \n"
                        "create the corresponding outputs.", argv[0]);
  const char *instem = argv[1];
  const char *tstem = argv[2];
  const char *sstem = argv[3];

  uint frame = 0; // frame number
  Image<float> prevlum; // luminance image of the previous frame, initially empty

  // setup some pyramid builders for the motion detection, using Reichardt motion detectors in 4 directions: NOTE: The
  // very small values below should be 0.0F, however, in our reference code, these are computed from an arbitrary motion
  // direction specified in degrees, using sin() and cos(), which apparently yields these almost-zero values:
  ReichardtPyrBuilder<float> dir0( 1.0F,         -0.0F,         Gaussian5,  90.0F);  // right
  ReichardtPyrBuilder<float> dir1( 6.12323e-17F, -1.0F,         Gaussian5, 180.0F);  // up
  ReichardtPyrBuilder<float> dir2(-1.0F,         -1.22465e-16F, Gaussian5, 270.0F);  // left
  ReichardtPyrBuilder<float> dir3(-1.83697e-16F,  1.0F,         Gaussian5, 360.0F);  // down

  try {
    // run the main loop forever until some exception occurs (e.g., no more input frames):
    while(true)
      {
        // read the next input frame from disk into a 24-bit RGB image:
        if (Raster::fileExists(sformat("%s%06u.png", instem, frame)) == false)
          { LINFO("Input frames exhausted -- Quit"); break; }

        Image< PixRGB<byte> > in = Raster::ReadRGB(sformat("%s%06u.png", instem, frame));

        LINFO("########## Processing frame %06u ##########", frame);

        // compute the luminance of the image, in float values:
        Image<float> lum = computeLuminance(in);

        // convert the RGB image to float color pixels:
        Image<PixRGB<float> > inf = in;

        // get R-G, B-Y, base-flicker components:
        Image<float> rg, by; getRGBY(inf, rg, by, 25.5F); // getRGBY() defined in Image/ColorOps.H
        Image<float> flick; if (prevlum.initialized()) flick = absDiff(lum, prevlum);

        // Compute image pyramids, center-surround differences across pairs of pyramid levels, and final resulting
        // conspicuity map for each channel. The saliency map will sum all the conspicuity maps, where each channel is
        // weighted by the number of feature maps it contains:
        const float w = 1.0F / float(num_feat_maps);

        // *** color channel:
        Image<float> chanc = normalizMAP(computeCMAP(by, Gaussian5, "blue-yellow", true) +
                                         computeCMAP(rg, Gaussian5, "red-green", true), Dims(0,0), "Color");

        Image<float> smap = chanc * (w / 2.0F);

        // *** flicker channel:
        if (flick.initialized()) {
          Image<float> chanf = normalizMAP(computeCMAP(flick, Gaussian5, "flicker", true), Dims(0,0), "Flicker");
          smap += chanf * w;
        }

        // *** intensity channel:
        const Image<float> chani = normalizMAP(computeCMAP(lum, Gaussian5, "intensity", true), Dims(0,0), "Intensity");

        smap += chani * w;

        // *** orientation channel:
        Image<float> chano = normalizMAP(computeCMAP(lum, Oriented9, "ori0", false, 0.0F), Dims(0,0), "ori0");
        chano += normalizMAP(computeCMAP(lum, Oriented9, "ori45", false, 45.0F), Dims(0,0), "ori45");
        chano += normalizMAP(computeCMAP(lum, Oriented9, "ori90", false, 90.0F), Dims(0,0), "ori90");
        chano += normalizMAP(computeCMAP(lum, Oriented9, "ori135", false, 135.0F), Dims(0,0), "ori135");

        smap += normalizMAP(chano, Dims(0,0), "Orientation") * (w / 4.0F);

        // *** and motion channel:
        ImageSet<float> d = dir0.build(lum, level_min, level_max + delta_max + 1);
        Image<float> chanm = normalizMAP(computeMMAP(d, "dir0"), Dims(0,0), "dir0");

        d = dir1.build(lum, level_min, level_max + delta_max + 1);
        chanm += normalizMAP(computeMMAP(d, "dir1"), Dims(0,0), "dir1");

        d = dir2.build(lum, level_min, level_max + delta_max + 1);
        chanm += normalizMAP(computeMMAP(d, "dir2"), Dims(0,0), "dir2");

        d = dir3.build(lum, level_min, level_max + delta_max + 1);
        chanm += normalizMAP(computeMMAP(d, "dir3"), Dims(0,0), "dir3");

        smap += normalizMAP(chanm, Dims(0,0), "Motion") * (w / 4.0F);

        // *** do one final round of spatial competition for salience (maxnorm) on the combined saliency map:
        smap = maxNormalize(smap, 0.0F, 2.0F, VCXNORM_MAXNORM);

        // finally convert saliency map values to nA of synaptic current:
        smap *= 1.0e-9F;

        float mi, ma; getMinMax(smap, mi, ma);
        LINFO("Saliency Map: final range [%f .. %f] nA", mi * 1.0e9F, ma * 1.0e9F);

        // create a composite image with input + saliency map + attention trajectory:
        Image<float> smapn = smap; inplaceNormalize(smapn, 0.0F, 255.0F);
        Image<PixRGB<byte> > traj = colGreyCombo(in, smapn, true, false);

        // write our results:
        Raster::WriteRGB(traj, sformat("%s%06u.png", tstem, frame));
        Raster::WriteFloat(smap, FLOAT_NORM_0_255, sformat("%s%06u.png", sstem, frame));
        Raster::WriteFloat(smap, FLOAT_NORM_PRESERVE, sformat("%s%06u.pfm", sstem, frame));

        // ready for next frame:
        prevlum = lum; ++frame;
      }
  } catch(...) { LERROR("Exception caught -- Quitting."); }

  return(0);
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
