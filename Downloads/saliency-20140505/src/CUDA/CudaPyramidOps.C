#include "CUDA/CudaImage.H"
#include "CUDA/CudaImageSet.H"
#include "Util/log.H"
#include "Image/PyramidOps.H"
#include "CudaPyramidOps.H"
#include "CUDA/CudaLowPass.H"
#include "CUDA/CudaKernels.H"
#include "CUDA/CudaMathOps.H"
#include "CUDA/CudaShapeOps.H"
#include "CUDA/CudaFilterOps.H"

// ######################################################################
// ##### Pyramid builder functions:
// ######################################################################

// ######################################################################
CudaImageSet<float> cudaBuildPyrGaussian(const CudaImage<float>& image,
                             int firstlevel, int depth, int filterSize)
{
  ASSERT(image.initialized());

  CudaImageSet<float> result(depth);

  if (0 >= firstlevel)
    result[0] = image;

  CudaImage<float> a = CudaImage<float>(image);
  for (int lev = 1; lev < depth; ++lev)
    {
      switch(filterSize)
        {
        case 5:
          a = cudaLowPass5yDec(cudaLowPass5xDec(a));
          break;
        default:
          a = cudaDecX(cudaLowPassX(filterSize, a));
          a = cudaDecY(cudaLowPassY(filterSize, a));
          break;
        }

      // NOTE: when lev < firstlevel, we leave result[lev] empty even
      // though it isn't any extra work to fill it in; that way, other
      // functions that see the result ImageSet might be able to gain
      // additional speed-ups by seeing that some of the images are
      // empty and avoid processing them further
      if (lev >= firstlevel)
        result[lev] = a;
    }

  return result;
}



// ######################################################################
CudaImageSet<float> cudaBuildPyrLaplacian(const CudaImage<float>& image, int firstlevel, int depth, int filterSize)
{
  ASSERT(image.initialized());

  CudaImageSet<float> result(depth);

  CudaImage<float> lpf = cudaLowPass(filterSize, image);

  if (0 >= firstlevel)
   result[0] = cudaSubtractImages(image,lpf);
  for (int lev = 1; lev < depth; ++lev)
    {
      const CudaImage<float> dec = cudaDecXY(lpf);
      lpf = cudaLowPass(filterSize, dec);

      if (lev >= firstlevel)
        result[lev] = cudaSubtractImages(dec,lpf);

    }

  return result;
}


// ######################################################################
CudaImageSet<float> cudaBuildPyrOrientedFromLaplacian(const CudaImageSet<float>& laplacian,
                                          int filterSize,
                                          float theta, float intens)
{
  int attenuation_width = -1;
  float spatial_freq = -1.0;

  switch (filterSize)
    {
    case 5: attenuation_width = 3; spatial_freq = M_PI_2; break;
    case 9: attenuation_width = 5; spatial_freq = 2.6; break;
    default:
      LFATAL("Filter size %d is not supported", filterSize);
    }

  // compute oriented filter from laplacian:
  CudaImageSet<float> result(laplacian.size());

  for (size_t lev = 0; lev < result.size(); ++lev)
    {
      if (laplacian[lev].initialized())
        {
          result[lev] = cudaOrientedFilter(laplacian[lev], spatial_freq,
                                       theta, intens);
          // attenuate borders that are overestimated due to filter trunctation:
          cudaInplaceAttenuateBorders(result[lev], attenuation_width);
        }
    }

  return result;
}

// ######################################################################
CudaImageSet<float> cudaBuildPyrOriented(const CudaImage<float>& image, int firstlevel, int depth, int filterSize,
                             float theta, float intens)
{
  ASSERT(image.initialized());

  const CudaImageSet<float> laplacian = cudaBuildPyrLaplacian(image, firstlevel, depth, filterSize);

  const CudaImageSet<float> orifilt =
    cudaBuildPyrOrientedFromLaplacian(laplacian, filterSize, theta,
                                  intens);

  CudaImageSet<float> result(orifilt.size());
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = orifilt[i];

  return result;
}



// ######################################################################
CudaImageSet<float> cudaBuildPyrLocalAvg(const CudaImage<float>& image, int depth)
{
  ASSERT(image.initialized());
  ASSERT(depth >= 1);

  CudaImageSet<float> result(depth);

  // only deepest level of pyr is filled with local avg:
  const int scale = int(pow(2.0, depth - 1));
  result[depth - 1] = cudaQuickLocalAvg(image, scale);

  return result;
}

// ######################################################################
CudaImageSet<float> cudaBuildPyrLocalAvg2x2(const CudaImage<float>& image, int depth)
{
  ASSERT(depth >= 0);

  if (depth == 0)
    return CudaImageSet<float>(0);

  CudaImageSet<float> result(depth);
  result[0] = image;

  for (int i = 1; i < depth; ++i)
    result[i] = cudaQuickLocalAvg2x2(result[i-1]);

  return result;
}

// ######################################################################
CudaImageSet<float> cudaBuildPyrLocalMax(const CudaImage<float>& image, int depth)
{
  ASSERT(image.initialized());
  ASSERT(depth >= 1);

  CudaImageSet<float> result(depth);

  // only deepest level of pyr is filled with local max:
  const int scale = int(pow(2.0, depth - 1));
  result[depth - 1] = cudaQuickLocalMax(image, scale);

  return result;
}


CudaImageSet<float> cudaBuildPyrGeneric(const CudaImage<float>& image,
                            int firstlevel, int depth,
                            const PyramidType typ, const float gabor_theta,
                            const float intens)
{
  switch(typ)
    {
    case Gaussian3: return cudaBuildPyrGaussian(image, firstlevel, depth, 3);
    case Gaussian5: return cudaBuildPyrGaussian(image, firstlevel, depth, 5);
    case Gaussian9: return cudaBuildPyrGaussian(image, firstlevel, depth, 9);

    case Laplacian5: return cudaBuildPyrLaplacian(image, firstlevel, depth, 5);
    case Laplacian9: return cudaBuildPyrLaplacian(image, firstlevel, depth, 9);

    case Oriented5: return cudaBuildPyrOriented(image, firstlevel, depth, 5, gabor_theta, intens);
    case Oriented9: return cudaBuildPyrOriented(image, firstlevel, depth, 9, gabor_theta, intens);

    case QuickLocalAvg: return cudaBuildPyrLocalAvg(image, depth);
    case QuickLocalMax: return cudaBuildPyrLocalMax(image, depth);

    default:
      LFATAL("Attempt to create Pyramid of unknown type %d", typ);
    }

  ASSERT(false);
  return CudaImageSet<float>(0); // "can't happen", but placate compiler
}



// ######################################################################
// CudaImageSet<float> cudaBuildPyrGabor(const CudaImageSet<float>& gaussianPyr,
//                               float angle, float filter_period,
//                                       float elongation, int size, int flags, MemoryPolicy mp, int dev)
// {
//   const double major_stddev = filter_period / 3.0;
//   const double minor_stddev = major_stddev * elongation;

//   // We have to add 90 to the angle here when constructing the gabor
//   // filter. That's because the angle used to build the gabor filter
//   // actually specifies the direction along which the grating
//   // varies. This direction is orthogonal to the the direction of the
//   // contours that the grating will detect.
//   const double theta = angle + 90.0f;

//   // In concept, we want to filter with four phases of the filter: odd
//   // on+off, and even on+off (that would be xox, oxo, xo, and ox). But
//   // since the on version just produces the negation of the off version,
//   // we can get the summed output of both by just taking the absolute
//   // value of the outputs of one of them. So we only need to convolve
//   // with the filter in two phases, then take the absolute value (i.e.,
//   // we're doing |xox| and |xo|).

//   CudaImage<float> g0 = cudaGaborFilter3(major_stddev, minor_stddev,
//                                          filter_period, 0.0f, theta, size, mp, dev);
//   CudaImage<float> g90 = cudaGaborFilter3(major_stddev, minor_stddev,
//                                           filter_period, 90.0f, theta, size, mp, dev);
//   LDEBUG("angle = %.2f, period = %.2f pix, size = %dx%d pix",
//          angle, filter_period, g0.getWidth(), g0.getHeight());

//   CudaImage<float> f0, f90;
//   CudaImageSet<float> result(gaussianPyr.size());

//   for (uint i = 0; i < gaussianPyr.size(); ++i)
//     {
//       // if the i'th level in our input is empty, then leave it empty
//       // in the output as well:
//       if (!gaussianPyr[i].initialized())
//         continue;
//       if (flags & DO_ENERGY_NORM)
//         {
//           CudaImage<float> temp = cudaEnergyNorm(in);
//           f0 = cudaOptConvolve(temp, g0);
//           f90 = cudaOptConvolve(temp, g90);
//         }
//       else
//         {
//           f0 = cudaOptConvolve(in, g0);
//           f90 = cudaOptConvolve(in, g90);
//         }

//       if (!(flags & NO_ABS))
//         {
//           f0 = cudaAbs(f0);
//           f90 = cudaAbs(f90);
//         }
//       result[i] = f0;
//       result[i] += f90;
//     }
//   return result;
// }


CudaImage<float> cudaCenterSurround(const CudaImageSet<float>& pyr,
                        const int lev1, const int lev2,
                        const bool absol,
                        const CudaImageSet<float>* clipPyr)
{
  ASSERT(lev1 >= 0 && lev2 >= 0);
  ASSERT(uint(lev1) < pyr.size() && uint(lev2) < pyr.size());

  const int largeLev = std::min(lev1, lev2);
  const int smallLev = std::max(lev1, lev2);

  if (clipPyr != 0 && clipPyr->isNonEmpty())
    {
      ASSERT((*clipPyr)[largeLev].getDims() == pyr[largeLev].getDims());
      ASSERT((*clipPyr)[smallLev].getDims() == pyr[smallLev].getDims());

      return
        ::cudaCenterSurround(pyr[largeLev]*(*clipPyr)[largeLev],
                             pyr[smallLev]*(*clipPyr)[smallLev],
                             absol);
    }
  else
    return ::cudaCenterSurround(pyr[largeLev], pyr[smallLev], absol);
}

void cudaCenterSurround(const CudaImageSet<float>& pyr,
                    const int lev1, const int lev2,
                    CudaImage<float>& pos, CudaImage<float>& neg,
                    const CudaImageSet<float>* clipPyr)
{
  ASSERT(lev1 >= 0 && lev2 >= 0);
  ASSERT(uint(lev1) < pyr.size() && uint(lev2) < pyr.size());

  const int largeLev = std::min(lev1, lev2);
  const int smallLev = std::max(lev1, lev2);

  if (clipPyr != 0 && clipPyr->isNonEmpty())
    {
      ASSERT((*clipPyr)[largeLev].getDims() == pyr[largeLev].getDims());
      ASSERT((*clipPyr)[smallLev].getDims() == pyr[smallLev].getDims());

      ::cudaCenterSurround(pyr[largeLev]*(*clipPyr)[largeLev],
                           pyr[smallLev]*(*clipPyr)[smallLev],
                           pos, neg);
    }
  else
    ::cudaCenterSurround(pyr[largeLev], pyr[smallLev], pos, neg);
}

CudaImage<float> cudaCenterSurroundSingleOpponent(const CudaImageSet<float>& cpyr,
                                      const CudaImageSet<float>& spyr,
                                      const int lev1, const int lev2,
                                      const bool absol,
                                      const CudaImageSet<float>* clipPyr)
{
  ASSERT(lev1 >= 0 && lev2 >= 0);
  ASSERT(uint(lev1) < cpyr.size() && uint(lev2) < cpyr.size());
  ASSERT(uint(lev1) < spyr.size() && uint(lev2) < spyr.size());

  const int largeLev = std::min(lev1, lev2);
  const int smallLev = std::max(lev1, lev2);

  if (clipPyr != 0 && clipPyr->isNonEmpty())
    {
      ASSERT((*clipPyr)[largeLev].getDims() == cpyr[largeLev].getDims());
      ASSERT((*clipPyr)[smallLev].getDims() == cpyr[smallLev].getDims());
      ASSERT((*clipPyr)[largeLev].getDims() == spyr[largeLev].getDims());
      ASSERT((*clipPyr)[smallLev].getDims() == spyr[smallLev].getDims());

      return
        ::cudaCenterSurround(cpyr[largeLev]*(*clipPyr)[largeLev],
                             spyr[smallLev]*(*clipPyr)[smallLev],
                             absol);
    }
  else
    return ::cudaCenterSurround(cpyr[largeLev], spyr[smallLev], absol);
}

void cudaCenterSurroundSingleOpponent(const CudaImageSet<float>& cpyr,
                                  const CudaImageSet<float>& spyr,
                                  const int lev1, const int lev2,
                                  CudaImage<float>& pos, CudaImage<float>& neg,
                                  const CudaImageSet<float>* clipPyr)
{
  ASSERT(lev1 >= 0 && lev2 >= 0);
  ASSERT(uint(lev1) < cpyr.size() && uint(lev2) < cpyr.size());
  ASSERT(uint(lev1) < spyr.size() && uint(lev2) < spyr.size());

  const int largeLev = std::min(lev1, lev2);
  const int smallLev = std::max(lev1, lev2);

  if (clipPyr != 0 && clipPyr->isNonEmpty())
    {
      ASSERT((*clipPyr)[largeLev].getDims() == cpyr[largeLev].getDims());
      ASSERT((*clipPyr)[smallLev].getDims() == cpyr[smallLev].getDims());
      ASSERT((*clipPyr)[largeLev].getDims() == spyr[largeLev].getDims());
      ASSERT((*clipPyr)[smallLev].getDims() == spyr[smallLev].getDims());

      ::cudaCenterSurround(cpyr[largeLev]*(*clipPyr)[largeLev],
                       spyr[smallLev]*(*clipPyr)[smallLev],
                       pos, neg);
    }
  else
    ::cudaCenterSurround(cpyr[largeLev], spyr[smallLev], pos, neg);
}

CudaImage<float> cudaCenterSurroundDiff(const CudaImageSet<float>& pyr1,
                            const CudaImageSet<float>& pyr2,
                            const int lev1, const int lev2,
                            const bool absol,
                            const CudaImageSet<float>* clipPyr)
{
  ASSERT(lev1 >= 0 && lev2 >= 0);
  ASSERT(uint(lev1) < pyr1.size() && uint(lev2) < pyr1.size());

  const int largeLev = std::min(lev1, lev2);
  const int smallLev = std::max(lev1, lev2);

  ASSERT(pyr1[largeLev].getDims() == pyr2[largeLev].getDims());
  ASSERT(pyr1[smallLev].getDims() == pyr2[smallLev].getDims());

  // compute differences between the two pyramids:
  const CudaImage<float> limg = pyr1[largeLev]-pyr2[largeLev];
  const CudaImage<float> simg = pyr1[smallLev]-pyr2[smallLev];

  if (clipPyr != 0 && clipPyr->isNonEmpty())
    {
      ASSERT((*clipPyr)[largeLev].getDims() == limg.getDims());
      ASSERT((*clipPyr)[smallLev].getDims() == simg.getDims());

      return ::cudaCenterSurround(limg*(*clipPyr)[largeLev],
                                  simg*(*clipPyr)[smallLev],
                                  absol);
    }
  else
    return ::cudaCenterSurround(limg, simg, absol);
}

void cudaCenterSurroundDiff(const CudaImageSet<float>& pyr1,
                        const CudaImageSet<float>& pyr2,
                        const int lev1, const int lev2,
                        CudaImage<float>& pos, CudaImage<float>& neg,
                        const CudaImageSet<float>* clipPyr)
{
  ASSERT(lev1 >= 0 && lev2 >= 0);
  ASSERT(uint(lev1) < pyr1.size() && uint(lev2) < pyr1.size());

  const int largeLev = std::min(lev1, lev2);
  const int smallLev = std::max(lev1, lev2);

  ASSERT(pyr1[largeLev].getDims() == pyr2[largeLev].getDims());
  ASSERT(pyr1[smallLev].getDims() == pyr2[smallLev].getDims());

  // compute differences between the two pyramids:
  const CudaImage<float> limg = pyr1[largeLev]-pyr2[largeLev];
  const CudaImage<float> simg = pyr1[smallLev]-pyr2[smallLev];

  if (clipPyr != 0 && clipPyr->isNonEmpty())
    {
      ASSERT((*clipPyr)[largeLev].getDims() == limg.getDims());
      ASSERT((*clipPyr)[smallLev].getDims() == simg.getDims());

      ::cudaCenterSurround(limg*(*clipPyr)[largeLev],
                       simg*(*clipPyr)[smallLev],
                       pos, neg);
    }
  else
    ::cudaCenterSurround(limg, simg, pos, neg);
}


CudaImage<float> cudaCenterSurroundDiffSingleOpponent(const CudaImageSet<float>& cpyr1,
                                          const CudaImageSet<float>& cpyr2,
                                          const CudaImageSet<float>& spyr1,
                                          const CudaImageSet<float>& spyr2,
                                          const int lev1, const int lev2,
                                          const bool absol,
                                          const CudaImageSet<float>* clipPyr)
{
  ASSERT(lev1 >= 0 && lev2 >= 0);
  ASSERT(uint(lev1) < cpyr1.size() && uint(lev2) < cpyr1.size());

  const int largeLev = std::min(lev1, lev2);
  const int smallLev = std::max(lev1, lev2);

  ASSERT(cpyr1[largeLev].getDims() == cpyr2[largeLev].getDims());
  ASSERT(cpyr1[smallLev].getDims() == cpyr2[smallLev].getDims());
  ASSERT(spyr1[largeLev].getDims() == spyr2[largeLev].getDims());
  ASSERT(spyr1[smallLev].getDims() == spyr2[smallLev].getDims());

  // compute differences between the two pyramids:
  const CudaImage<float> limg = cpyr1[largeLev]-cpyr2[largeLev];
  const CudaImage<float> simg = spyr1[smallLev]-spyr2[smallLev];

  if (clipPyr != 0 && clipPyr->isNonEmpty())
    {
      ASSERT((*clipPyr)[largeLev].getDims() == limg.getDims());
      ASSERT((*clipPyr)[smallLev].getDims() == simg.getDims());

      return ::cudaCenterSurround(limg*(*clipPyr)[largeLev],
                                  simg*(*clipPyr)[smallLev],
                                  absol);
    }
  else
    return ::cudaCenterSurround(limg, simg, absol);
}


void cudaCenterSurroundDiffSingleOpponent(const CudaImageSet<float>& cpyr1,
                                      const CudaImageSet<float>& cpyr2,
                                      const CudaImageSet<float>& spyr1,
                                      const CudaImageSet<float>& spyr2,
                                      const int lev1, const int lev2,
                                      CudaImage<float>& pos, CudaImage<float>& neg,
                                      const CudaImageSet<float>* clipPyr)
{
  ASSERT(lev1 >= 0 && lev2 >= 0);
  ASSERT(uint(lev1) < cpyr1.size() && uint(lev2) < cpyr1.size());

  const int largeLev = std::min(lev1, lev2);
  const int smallLev = std::max(lev1, lev2);

  ASSERT(cpyr1[largeLev].getDims() == cpyr2[largeLev].getDims());
  ASSERT(cpyr1[smallLev].getDims() == cpyr2[smallLev].getDims());
  ASSERT(spyr1[largeLev].getDims() == spyr2[largeLev].getDims());
  ASSERT(spyr1[smallLev].getDims() == spyr2[smallLev].getDims());

  // compute differences between the two pyramids:
  const CudaImage<float> limg = cpyr1[largeLev]-cpyr2[largeLev];
  const CudaImage<float> simg = spyr1[smallLev]-spyr2[smallLev];

  if (clipPyr != 0 && clipPyr->isNonEmpty())
    {
      ASSERT((*clipPyr)[largeLev].getDims() == limg.getDims());
      ASSERT((*clipPyr)[smallLev].getDims() == simg.getDims());

      ::cudaCenterSurround(limg*(*clipPyr)[largeLev],
                           simg*(*clipPyr)[smallLev],
                           pos, neg);
    }
  else
    ::cudaCenterSurround(limg, simg, pos, neg);
}
