/*!@file MBARI/FOEestimator.C a class for estimating the focus of expansion
  in a video
 */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/FOEestimator.C $
// $Id: FOEestimator.C 7293 2006-10-20 18:49:55Z rjpeters $
//

#include "MBARI/FOEestimator.H"

#include "Image/CutPaste.H"  // for crop()
#include "Image/ImageSet.H"
#include "Image/PyramidOps.H"

#include <vector>

// ######################################################################
FOEestimator::FOEestimator(int numAvg, int pyramidLevel)
  : itsPyrLevel(pyramidLevel),
    itsFrames(3),
    itsXvectors(numAvg),
    itsYvectors(numAvg)
{ }

// ######################################################################
Vector2D FOEestimator::updateFOE(const Image<byte>& img,
                                 const Rectangle region)
{
  return (updateFOE(crop(img,region)) +
          Vector2D(region.left(),region.top()));
}

// ######################################################################
Vector2D FOEestimator::updateFOE(const Image<byte>& img)
{
  // subsample the image to the desired level
  ImageSet<byte> iset = buildPyrGaussian(img,0,itsPyrLevel+1,3);
  itsFrames.push_back(iset[itsPyrLevel]);

  // need at least three frames to compute optical flow
  if (itsFrames.size() < 3) return Vector2D();

  int w = itsFrames.back().getWidth();
  int h = itsFrames.back().getHeight();

  // get the iterators ready to step through the three images
  // at times (t-1), t, and (t+1)
  Image<byte>::const_iterator it,tm,tp,xm,xp,ym,yp;
  it = itsFrames[1].begin();
  xm = it + w;
  xp = it + w + 2;
  ym = it + 1;
  yp = it + w + w + 1;
  tm = itsFrames[0].begin() + w + 1;
  tp = itsFrames[2].begin() + w + 1;

  // prepare the vectors for the results
  Image<float> vx(w-2,1,ZEROS);
  Image<float> vy(h-2,1,ZEROS);
  std::vector<int> cx(w-2,0), cy(h-2,0);

  // compute optical flow and sum up the x components over y, and the
  // y components over x
  for (int y = 0; y < h-2; ++y)
    {
      for (int x = 0; x < w-2; ++x)
        {
          // x component of optical flow
          if (*xp != *xm)
            {
              vx[x] += (float(*tm)-float(*tp)) / (float(*xp) - float(*xm));
              cx[x]++;
            }

          // y component of optical flow
          if (*yp != *ym)
            {
              vy[y] += (float(*tm)-float(*tp)) / (float(*yp) - float(*ym));
              cy[y]++;
            }
          ++tm; ++tp; ++xm; ++xp; ++ym; ++yp;
        }

      // divide the y guys by the number of contributions
      if (cy[y] > 0) vy[y] /= cy[y];

      // offset to jump to the next line
      tm += 2; tp += 2;
      xm += 2; xp += 2;
      ym += 2; yp += 2;
    }

  // divide the x guys by the number of contributions
  for (int x = 0; x < w-2; ++x)
      if (cx[x] > 0) vx[x] /= cx[x];

  // store the vectors in the cache to generate averages
  itsXvectors.push_back(vx);
  itsYvectors.push_back(vy);

  // now do the linear regression for the x and y directions over the mean
  float x0 = getZeroCrossing(itsXvectors.mean());
  float y0 = getZeroCrossing(itsYvectors.mean());

  // need to scale the coordinates according to the subsampling done
  float pw2 = pow(2,itsPyrLevel);
  itsFOE.reset(pw2*(x0+1),pw2*(y0+1));

  return itsFOE;
}



// ######################################################################
float FOEestimator::getZeroCrossing(const Image<float>& vec)
{
  float N = vec.getWidth();
  float sx = N * (N - 1) / 2;
  float sxx = N * (N - 1) * (2*N - 1) / 6;
  float sy = 0.0F, sxy = 0.0F;
  for (int i = 0; i < N; ++i)
    {
      sy += vec[i];
      sxy += i*vec[i];
    }
  return (sx * sxy - sxx * sy) / (N * sxy - sx * sy);
}


// ######################################################################
Vector2D FOEestimator::getFOE()
{
  return itsFOE;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
