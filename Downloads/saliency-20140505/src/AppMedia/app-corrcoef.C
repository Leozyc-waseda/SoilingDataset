/*! @file AppMedia/app-corrcoef.C simple app to compute the
  correlation coefficient between two images, and also print basic
  first order statistics for each image */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-corrcoef.C $
// $Id: app-corrcoef.C 6263 2006-02-18 00:07:18Z rjpeters $

#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Range.H"
#include "Raster/Raster.H"
#include <cstdio>

#ifdef HAVE_FENV_H
#include <fenv.h>
#endif

int main(int argc, char** argv)
{
  if (argc != 3)
    {
      printf("usage: %s image1 image2\n", argv[0]);
      return 1;
    }

  const Image<float> img1 = Raster::ReadFloat(argv[1]);
  const Image<float> img2 = Raster::ReadFloat(argv[2]);

#ifdef HAVE_FEENABLEEXCEPT
  feenableexcept(FE_DIVBYZERO|FE_INVALID);
#endif

  const double rsq = corrcoef(img1, img2);

  const Range<float> r1 = rangeOf(img1);
  const Range<float> r2 = rangeOf(img2);

  printf("mean(%s)=%e\n", argv[1], mean(img1));
  printf("mean(%s)=%e\n", argv[2], mean(img2));
  printf("stdev(%s)=%e\n", argv[1], stdev(img1));
  printf("stdev(%s)=%e\n", argv[2], stdev(img2));
  printf("range(%s)=[%e .. %e]\n", argv[1], r1.min(), r1.max());
  printf("range(%s)=[%e .. %e]\n", argv[2], r2.min(), r2.max());
  printf("corrcoef(%s,%s)=%e\n", argv[1], argv[2], sqrt(rsq));
}
