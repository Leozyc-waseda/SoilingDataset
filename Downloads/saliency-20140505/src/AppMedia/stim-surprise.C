/*!@file AppMedia/stim-surprise.C generate surprising stimuli */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/stim-surprise.C $
// $Id: stim-surprise.C 9412 2008-03-10 23:10:15Z farhan $

#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"
#include "Util/MathFunctions.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include <vector>

#define WIDTH 256
#define NFRAME 300
#define DX 32
#define DTHETA 0.1
#define RAD 8

int main(const int argc, const char **argv)
{
  initRandomNumbers();
  Image<byte> img(WIDTH, WIDTH, NO_INIT);

  if (argc < 2) LFATAL("USAGE: stim_surprise <type>");

  // arrays of precessing dots and one is special in there?
  if (strcmp(argv[1], "dots") == 0)
    {
      // get random centers:
      std::vector<Point2D<int> > center; std::vector<double> phi;
      for (int j = DX/2; j < WIDTH; j += DX)
        for (int i = DX/2; i < WIDTH; i += DX)
          {
            center.push_back(Point2D<int>(i + int(5.0 * (randomDouble()-0.5)),
                                     j + int(5.0 * (randomDouble()-0.5))));
            phi.push_back(M_PI * randomDouble());
          }

      double theta = 0.0, thetac = 0.0;;
      for (int k = 0; k < NFRAME; k ++)
        {
          img.clear();

          for (unsigned int n = 0; n < center.size(); n ++)
            {
              double t = theta, rx = RAD, ry = RAD, p = phi[n];
              byte col = 255;
              if (n == 27) t = thetac;
              //if (n == 27) { rx = 0; ry = 0; if ((k/6) & 1) col = 0; }

              int ii = int(center[n].i + rx * cos(t + p));
              int jj = int(center[n].j + ry * sin(t + p));
              drawDisk(img, Point2D<int>(ii, jj), 4, col);
            }

          Raster::WriteRGB(img, sformat("frame%06d.ppm", k));
          theta += DTHETA; //thetac -= DTHETA; // += 3.67 * DTHETA;
        }
    }
  // random noise?
  else if (strcmp(argv[1], "rnd") == 0)
    {
      for (int k = 0; k < NFRAME; k ++)
        {
          for (int j = 0; j < WIDTH; j ++)
            for (int i = 0; i < WIDTH; i ++)
              {

                double d2 = (i-127)*(i-127) + (j-127)*(j-127);
                double coeff = exp(-d2 / 400.0) * 0.5;

                img.setVal(i, j, 128 + int(256.0 * (1.0-coeff) * randomDouble()) -
                           int(128.0 * (1.0-coeff)));
              }
          Raster::WriteRGB(img, sformat("frame%06d.ppm", k));
        }
    }
  // random blinking?
  else if (strcmp(argv[1], "blink") == 0)
    {
      // get random centers:
      std::vector<Point2D<int> > center; std::vector<int> phi;
      const int period = 20;
      for (int j = DX/2; j < WIDTH; j += DX)
        for (int i = DX/2; i < WIDTH; i += DX)
          {
            center.push_back(Point2D<int>(i + int(5.0 * (randomDouble()-0.5)),
                                     j + int(5.0 * (randomDouble()-0.5))));
            phi.push_back(randomUpToNotIncluding(period));
          }

      for (int k = 0; k < NFRAME; k ++)
        {
          img.clear();

          for (unsigned int n = 0; n < center.size(); n ++)
            {
              byte col;
              if (n == 27)
                {
                  if (((k + phi[n]) & 4) < 2) col = 255; else col = 0;
                }
              else
                {
                  if (((k + phi[n]) % period) < period/2) col = 255;
                  else col = 0;
                }
              drawDisk(img, center[n], 4, col);
            }

          Raster::WriteRGB(img, sformat("frame%06d.ppm", k));

        }
    }
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
