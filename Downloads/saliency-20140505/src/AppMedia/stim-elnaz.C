/*!@file AppMedia/stim-surprise.C generate surprising stimuli */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/stim-elnaz.C $
// $Id: stim-elnaz.C 12962 2010-03-06 02:13:53Z irock $

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

#define DOT_RAD 4
#define DOT_SPEED_PIXEL_PER_FRAME 2

#define DEATH_RATE_PERCENT 0

#define DOT_LIFE_TIME 200

int main(const int argc, const char **argv)
{
  initRandomNumbers();
  Image<byte> img(WIDTH, WIDTH, NO_INIT);

  if (argc < 2) LFATAL("USAGE: stim_surprise <type>");

 // arrays of precessing dots and one is special in there?

  if (strcmp(argv[1], "foe") == 0 || strcmp(argv[1], "foc") == 0)
    {

        img.clear(128);

        Raster::WriteRGB(img, "greyframe.ppm");

        int direction=0;
        if (strcmp(argv[1], "foe") == 0){
                direction=1;
        }
        if (strcmp(argv[1], "foc") == 0){
                direction=-1;
        }

//        int speed=2; // pixel per frame
        Point2D<int> foe= Point2D<int>(0 , 0);
      // get random centers:
      std::vector<Point2D<int> > center;
std::vector<Point2D<int> > temp;
 std::vector<double> phi;
      for (int j = DX/2; j < WIDTH; j += DX)
        for (int i = DX/2; i < WIDTH; i += DX)
          {
            center.push_back(Point2D<int>(i + int(5.0 * (randomDouble()-0.5)), j + int(5.0 * (randomDouble()-0.5))));
            phi.push_back(M_PI * randomDouble());
          }

      double theta = 0.0;

        int count=0;
      for (int k = 0; k < NFRAME; k ++)
        {
          img.clear(128 );

          for (unsigned int n = 0; n < center.size(); n ++)
            {

              byte col_in_white = 255;
              byte col_out_black = 0;
                // check if they have gone out
                if(direction>0) //foe
                {
                        int ii = int(center[n].i + direction*(center[n].i-foe.i)*(k*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        if(ii > WIDTH || ii< 0){
                                ii = int(center[n].i + direction*(center[n].i-foe.i)*(theta-count*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        }
                        //center[n].i=ii;
                              int jj = int(center[n].j + direction*(center[n].j-foe.j)*(k*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        if(jj > WIDTH || jj< 0){
                                jj = int(center[n].j + direction*(center[n].j-foe.j)*(theta-count*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        }
                        //center[n].j=jj;

                        temp.push_back(Point2D<int>(ii, jj));

              drawDisk(img, Point2D<int>(ii, jj), DOT_RAD*2, col_out_black);
              drawDisk(img, Point2D<int>(ii, jj), DOT_RAD, col_in_white);


                }

                if(direction<0) //foc
                {
                        int ii = int(center[n].i + direction*(center[n].i-foe.i)*(k*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        if(ii > WIDTH || ii< 0){
                                ii = int(center[n].i + direction*(center[n].i-foe.i)*(theta-count*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        }
                        //center[n].i=ii;
                              int jj = int(center[n].j + direction*(center[n].j-foe.j)*(k*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        if(jj > WIDTH || jj< 0){
                                jj = int(center[n].j + direction*(center[n].j-foe.j)*(theta-count*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        }
                        //center[n].j=jj;

              drawDisk(img, Point2D<int>(ii, jj), DOT_RAD*2, col_out_black);
              drawDisk(img, Point2D<int>(ii, jj), DOT_RAD, col_in_white);


                }

            }



          Raster::WriteRGB(img, sformat("frame%06d.ppm", k));
          theta += DOT_SPEED_PIXEL_PER_FRAME; //thetac -= DTHETA; // += 3.67 * DTHETA;
        count++;
        if(count>DOT_LIFE_TIME)
                count=0;
        }
    }
  // random noise?

  // arrays of precessing dots and one is special in there?
   else if (strcmp(argv[1], "dots") == 0)
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
