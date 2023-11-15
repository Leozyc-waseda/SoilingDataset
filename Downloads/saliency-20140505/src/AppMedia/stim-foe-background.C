/*!@file AppMedia/stim-surprise.C generate surprising stimuli */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/stim-foe-background.C $
// $Id: stim-foe-background.C 12962 2010-03-06 02:13:53Z irock $

#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"
#include "Util/MathFunctions.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include <vector>

#define WIDTH 1280
#define HEIGHT 720
#define NFRAME 500
#define DX 72
#define DX_DOT_OVERLAP 3
#define DTHETA 0.1
#define RAD 8

#define DOT_RAD 5
#define DOT_SPEED_PIXEL_PER_FRAME 2

#define DOT_DEATH_RATE_PERCENT 0

#define DOT_LIFE_TIME 1000
#define RATIO 100


template <class T>
void drawDisk2(Image<T>& dst, const Point2D<int>& center,
              const float radius, const T value)
{
  if (radius == 1)
    {
      if (dst.coordsOk(center)) dst.setVal(center.i + dst.getWidth() * center.j, value);
      return;
    }

  for (int y = -int(radius); y <= int(radius); ++y)
    {
      int bound = int(sqrtf(float(squareOf(radius) - squareOf(y))));
      for (int x = -bound; x <= bound; ++x)
        if (dst.coordsOk(x + center.i, y + center.j))
          dst.setVal(x + center.i, y + center.j, value);
    }
}


int main(const int argc, const char **argv)
{
  initRandomNumbers();
  Image<PixRGB<byte> > img(WIDTH, HEIGHT, NO_INIT);

  if (argc < 2) LFATAL("USAGE: stim_surprise <type>");


 // arrays of precessing dots and one is special in there?

  if (strcmp(argv[1], "foe") == 0 || strcmp(argv[1], "foc") == 0)
    {

        img.clear(PixRGB<byte>(128,128,128));
        //Raster::WriteRGB(img, "greyframe.ppm");
        int direction=0;
        if (strcmp(argv[1], "foe") == 0){
                direction=1;
        }
        if (strcmp(argv[1], "foc") == 0){
                direction=-1;
        }
        Point2D<float> foe= Point2D<float>(float(float(WIDTH)*9/10) , float(float(HEIGHT)*9/10));
      // get random centers:
      std::vector<Point2D<float> > center;
      std::vector<int > center_life;
      std::vector<float> center_depth;

      std::vector<Point2D<float> > temp;
      std::vector<int > temp_life;
      std::vector<float> temp_depth;
      //initialize
      std::vector<double> phi;
      for (int j = DX/2; j < HEIGHT; j += DX)
        for (int i = DX/2; i < WIDTH; i += DX)
          {
            center.push_back(Point2D<float>(i + float(10.0 * (randomDouble()-5.0)), j + float(10.0 * (randomDouble()-5.0))));
            center_life.push_back(4);
            phi.push_back(M_PI * randomDouble());
          }

        int count=0;
        unsigned int count_dots_disregard=0;
      for (int k = 0; k < NFRAME; k ++){
        PixRGB<byte> col_in_white = PixRGB<byte>(256,256,256);
        PixRGB<byte> col_out_black = PixRGB<byte>(0,0,0);
        img.clear(PixRGB<byte>(128,128,128));
        count_dots_disregard=0;
          for (unsigned int n = 0; n < center.size(); n ++)
          {
            // check if they have gone out
                if(direction>0) //foe
                {
                        float ii,jj;
                        if(k==0){
                                ii = float(center[n].i  );
                                      jj = float(center[n].j  );
                        }else{
                                ii = float(center[n].i + direction*float(center[n].i-foe.i)*float(DOT_SPEED_PIXEL_PER_FRAME)/RATIO );
                                      jj = float(center[n].j + direction*float(center[n].j-foe.j)*float(DOT_SPEED_PIXEL_PER_FRAME)/RATIO );
                                if(ii==center[n].i){
                                        if( (center[n].i-foe.i)<0 )
                                                ii=float(ii-1);
                                        else
                                                ii=float(ii+1);
                                }
                                if(jj==center[n].j){
                                        jj=float(jj+1);
                                if( (center[n].j-foe.j)<0 )
                                        jj=float(jj-1);
                                else
                                        jj=float(jj+1);
                                }
                        }
                        if(ii < WIDTH && ii> 0 && jj< HEIGHT && jj> 0 && center_life[n]<DOT_LIFE_TIME ){
                                temp.push_back(Point2D<float>(ii, jj));
                                temp_life.push_back(center_life[n]+1);
                        }else{
                                count_dots_disregard++;
                        }

                }

                if(direction<0) //foc
                {
                        float ii = float(center[n].i + direction*(center[n].i-foe.i)*(k*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        if(ii > WIDTH || ii< 0){
                                //ii = float(center[n].i + direction*(center[n].i-foe.i)*(theta-count*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        }
                        //center[n].i=ii;
                              float jj = float(center[n].j + direction*(center[n].j-foe.j)*(k*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        if(jj > HEIGHT || jj< 0){
                                //jj = float(center[n].j + direction*(center[n].j-foe.j)*(theta-count*DOT_SPEED_PIXEL_PER_FRAME)/DOT_LIFE_TIME );
                        }
                        //center[n].j=jj;

                              drawDisk(img, Point2D<int>(ii, jj), DOT_RAD*2, col_out_black);
                              drawDisk(img, Point2D<int>(ii, jj), DOT_RAD, col_in_white);
                }

            }


        for (unsigned int a = 0; a < count_dots_disregard; a ++){
                float temp_i,temp_j;
                //bool good=false;
                temp_i=float(WIDTH * (randomDouble()));
                temp_j=float(HEIGHT * (randomDouble()));
                /*while(!good){
                        temp_i=int(WIDTH * (randomDouble()));
                        temp_j=int(WIDTH * (randomDouble()));
                        bool overlap=false;
                        for (unsigned int n = 0; n < temp.size(); n ++){
                                if(abs(temp_i-temp[n].i)<DX_DOT_OVERLAP || abs(temp_j-temp[n].j)<DX_DOT_OVERLAP){
                                        overlap=true;
                                        break;
                                }
                        }
                        if(!overlap){
                                good=true;
                        }
                }*/
                temp.push_back(Point2D<float>(temp_i , temp_j ));
                temp_life.push_back(0);
        }

// Draw dots
        center.clear();
        center_life.clear();
         for (unsigned int n = 0; n < temp.size(); n ++){
                center.push_back(temp[n]);
                center_life.push_back(temp_life[n]);
                if(1 ){ //float(temp_life[n]/10)< float(DOT_RAD*2)
                        drawDisk2(img, Point2D<int>(int(temp[n].i), int(temp[n].j) ), float(float(temp_life[n])/15*2), col_out_black);
                        drawDisk2(img, Point2D<int>(int(temp[n].i), int(temp[n].j) ), float(temp_life[n])/15, col_in_white);
                }
                else if(temp_life[n]> 99999){ //(DOT_LIFE_TIME-DOT_RAD)
                        drawDisk2(img, Point2D<int>(int(temp[n].i), int(temp[n].j) ), (DOT_LIFE_TIME-temp_life[n])*2, col_out_black);
                        drawDisk2(img, Point2D<int>(int(temp[n].i), int(temp[n].j) ), DOT_LIFE_TIME-temp_life[n], col_in_white);
                }else{
                        drawDisk2(img, Point2D<int>(int(temp[n].i), int(temp[n].j) ), float(DOT_RAD*2), col_out_black);
                        drawDisk2(img, Point2D<int>(int(temp[n].i), int(temp[n].j) ), float(DOT_RAD), col_in_white);
                }
        }
        temp.clear();
        temp_life.clear();
          Raster::WriteRGB(img, sformat("/home2/tmp/u/elno/research/exp1/stim/bgMovies/foe/frames/frame%06d.ppm", k));
          //theta += DOT_SPEED_PIXEL_PER_FRAME; //thetac -= DTHETA; // += 3.67 * DTHETA;
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
              PixRGB<byte> col= PixRGB<byte>(256,256,256);
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
              PixRGB<byte> col= PixRGB<byte>(256,256,256);
              if (n == 27)
                {
                  if (((k + phi[n]) & 4) < 2) col = PixRGB<byte>(256,256,256); else col = PixRGB<byte>(0,0,0);
                }
              else
                {
                  if (((k + phi[n]) % period) < period/2) col = PixRGB<byte>(256,256,256);
                  else col = PixRGB<byte>(0,0,0);
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
