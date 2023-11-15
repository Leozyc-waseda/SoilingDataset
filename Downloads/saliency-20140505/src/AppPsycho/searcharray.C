/*!@file AppPsycho/searcharray.C create a randomized search array from two image patches */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/searcharray.C $
// $Id: searcharray.C 7157 2006-09-15 07:55:58Z itti $

#include "Util/Assert.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Util/log.H"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <unistd.h>

//! Spacing between elements as a factor of their size:
#define FACTOR 1.2

void image_patch(const Image< PixRGB<byte> >& patch, const int ti,
                 const int tj, Image< PixRGB<byte> >& image,
                 const double alpha,  Image<byte>& targets, bool do_target);


/*! This program generates a randomized search array (and associated
  target mask) from two image patches (for target and distractors). */

int main(const int argc, const char **argv)
{
  if (argc != 7)
    {
      std::cerr<<"USAGE: searcharray <patch1.ppm> <patch2.ppm> <w> <alpha> ";
      std::cerr<<"<noise> <result>"<<std::endl;
      exit(1);
    }
  int w = atoi(argv[3]); float alpha = atof(argv[4]);
  float noise = atof(argv[5]);
  initRandomNumbers();

  // read in the patches:
  Image<PixRGB<byte> > patch1 = Raster::ReadRGB(argv[1]);
  Image<PixRGB<byte> > patch2 = Raster::ReadRGB(argv[2]);

  int pw = patch1.getWidth(), ph = patch1.getHeight();
  ASSERT(pw == patch2.getWidth() && ph == patch2.getHeight());

  // initialize results:
  Image<PixRGB<byte> > image(w, w, ZEROS);
  Image<byte> targets(w, w, ZEROS);

  int ti = 1 + int(randomDouble() * floor(w / (FACTOR * pw) * 0.999 - 2.0));
  int tj = 1 + int(randomDouble() * floor(w / (FACTOR * ph) * 0.999 - 2.0));
  LINFO("-- TARGET AT (%d, %d)", ti, tj);

  for (int j = 0; j < int(w / (FACTOR * ph)); j++)
    for (int i = 0; i < int(w / (FACTOR * pw)); i++)
      if (i == ti && j == tj)
        image_patch(patch2, ti, tj, image, alpha, targets, 1);
      else
        image_patch(patch1, i, j, image, alpha, targets, 0);

  /* add noise */
  Image< PixRGB<byte> >::iterator iptr = image.beginw(), stop = image.endw();
  while(iptr != stop)
    {
      if (randomDouble() <= noise)
        {
          if (randomDouble() >= 0.5) iptr->setRed(255); else iptr->setRed(0);
          if (randomDouble() >= 0.5) iptr->setGreen(255); else iptr->setGreen(0);
          if (randomDouble() >= 0.5) iptr->setBlue(255); else iptr->setBlue(0);
        }
      iptr ++;
    }

  Raster::WriteRGB(image, argv[6], RASFMT_PNM);
  Raster::WriteGray(targets, argv[6], RASFMT_PNM);

  return 0;
}

// ######################################################################
void image_patch(const Image< PixRGB<byte> >& patch, const int ti,
                 const int tj, Image< PixRGB<byte> >& image,
                 const double alpha,  Image<byte>& targets, bool do_target)
{
  int pw = patch.getWidth(), ph = patch.getHeight();
  int w = image.getWidth();

  int jitx = int(randomDouble() * (FACTOR - 1.0) * pw);
  int jity = int(randomDouble() * (FACTOR - 1.0) * ph);

  float jita = float(alpha * 3.14159 / 180.0 * (randomDouble() - 0.5) * 2.0);
  int offset = int(w - floor(w / (pw * FACTOR)) * (pw * FACTOR)) / 2;

  PixRGB<byte> zero(0, 0, 0);
  int px = 0, py = 0;

  for (double y = int(tj * ph * FACTOR); y < int(tj * ph * FACTOR + ph); y ++)
    {
      for (double x = int(ti * pw * FACTOR); x < int(ti * pw * FACTOR + pw); x ++)
        {
          int x2 = int(x + jitx + offset);
          int y2 = int(y + jity + offset);

          /* Shifting back and forth the center of rotation.*/
          double px2 = px - pw / 2.0F;
          double py2 = py - ph / 2.0F;

          float px3 = float(cos(jita) * px2 + sin(jita) * py2 + pw / 2.0F);
          float py3 = float(-sin(jita) * px2 + cos(jita) * py2 + pw / 2.0F);

          if (px3 < 0 || px3 >= pw || py3 < 0 || py3 >= ph )
            image.setVal(x2, y2, zero);
          else
            {
              image.setVal(x2, y2, patch.getValInterp(px3, py3));
              if (do_target)
                {
                  if (patch.getVal(int(px3), int(py3)) == zero)
                    targets.setVal(x2, y2, 0);
                  else
                    targets.setVal(x2, y2, 255);
                }
            }
          px ++;
        }
      py ++;
      px = 0;
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
