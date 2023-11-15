/*
 *  test-BOWN_Mod.C
 *
 *
 *  Created by Randolph Voorhies
 *
 */



#include "Component/ModelManager.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Media/MediaOpts.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "GUI/XWinManaged.H"
#include "Image/ImageSet.H"
#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"
#include "Image/Kernels.H"
#include "Image/Normalize.H"
#include "Image/SimpleFont.H"
#include "rutz/trace.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#define PI 3.14159265F
#define THETA_LEVELS 24
#define MAP_SIZE 100
#define MOD_WEIGHT_WIDTH MAP_SIZE/6
#define OUTPUT_RES 500
#define OUT_SCALE_FACT OUTPUT_RES/MAP_SIZE
#define BRIGHTNESS 10

using namespace std;

float beta(Point2D<int> i, Point2D<int> j);
float M(Image<float> WeightKernel, float theta, Point2D<int> i, Point2D<int> att);
//Data Set 1: A single bistable vertical line
void initDataSet1(vector < Image<float> > &V2,  Point2D<int> &attentionPoint, float &k);
//Data Set 2: A single bistable PI/4 line
void initDataSet2(vector< Image<float> > &V2, Point2D<int> &attentionPoint, float &k);
//Data Set 3: A single bistable square
void initDataSet3(vector< Image<float> > &V2, Point2D<int> &attentionPoint, float &k);
//Data Set 4: A set of 5 ambiguously owned squares, sharing borders
void initDataSet4(vector< Image<float> > &V2, Point2D<int> &attentionPoint, float &k);
//Data Set 5: A rough imitation of a "Rubins Vase,"
void initDataSet5(vector< Image<float> > &V2, Point2D<int> &attentionPoint, float &k);

template <class T>
void drawHalfLine(Image<T>& dst,
    const Point2D<int>& pos, float ori, float len, const T col,
                  const int rad = 1);

template <class T>
void flipVertical(Image <T> &Image);


//Quick and dirty interface button
class button {
public:
  button() {
    width = int(.5*OUTPUT_RES-.1*OUTPUT_RES);
    height = int(.5*width);
    rect=Rectangle(topLeft, Dims(width, height));
  }
  button(const char* t, Point2D<int> pos) {
    strncpy(text,t,30);
    width = int(.5*OUTPUT_RES-.2*OUTPUT_RES);
    height = width/3;

    topLeft = pos;

    rect=Rectangle(topLeft, Dims(width, height));
  }

  bool inBounds(Point2D<int> click) {
    if(click.i > topLeft.i && click.i < topLeft.i+width && click.j > topLeft.j && click.j < topLeft.j+height)
      return true;
    else
      return false;
  }

  void draw(Image<PixRGB<byte> > &img) {
    drawRect(img, rect, PixRGB<byte>(255,0,0));
    writeText(img, topLeft+height/2-8, text, PixRGB<byte>(255,255,255), PixRGB<byte>(255,0,0), SimpleFont::FIXED(8), true);
  }

  char text[30];
  int width;
  int height;
  Point2D<int> topLeft;
  Rectangle rect;
};

int main(int argc, char* argv[]) {


  ModelManager manager("Border Ownership Modulator");

  //Start the model manager
  manager.start();

  //The Main Display Window
  rutz::shared_ptr<XWinManaged> window1;
  //rutz::shared_ptr<XWinManaged> window2;


  window1.reset(new XWinManaged(Dims(int(OUTPUT_RES*1.5),OUTPUT_RES), 0, 0, "Border Ownership"));
  //window2.reset(new XWinManaged(Dims(OUTPUT_RES*1.5,OUTPUT_RES), 0, 0, "Border Ownership"));


  //The precomputed gaussian kernel, normalized to 0.0 < value < 1.0
  Image<float> gaussianKernel = gaussian2D<float>(MOD_WEIGHT_WIDTH);
  normalizeFloat(gaussianKernel, FLOAT_NORM_0_255);
  gaussianKernel /= 255.0F;

  //An aggregation of V2 orientations showing the effects of attentional modulation
  Image< PixRGB<byte> > outputImage = Image< PixRGB<byte> >(Dims(OUTPUT_RES,OUTPUT_RES), ZEROS);
  Image< PixRGB<byte> > panelImage = Image< PixRGB<byte> >(Dims(OUTPUT_RES/2,OUTPUT_RES), ZEROS);

  //The V2 vector is an array of matrices that represents the hypercolumns in V2
  vector< Image<float> > V2;
  V2.resize(THETA_LEVELS, Image<float>(Dims(MAP_SIZE, MAP_SIZE), ZEROS));

  Point2D<int> attentionPoint; //The point of top down attention

  button but_Data1("Load Data Set 1", Point2D<int>(50, 30));
  button but_Data2("Load Data Set 2", Point2D<int>(50, 30+1*but_Data1.height));
  button but_Data3("Load Data Set 3", Point2D<int>(50, 30+2*but_Data1.height));
  button but_Data4("Load Data Set 4", Point2D<int>(50, 30+3*but_Data1.height));
  button but_Data5("Load Data Set 5", Point2D<int>(50, 30+4*but_Data1.height));
  button but_Quit("Quit",             Point2D<int>(50, OUTPUT_RES-but_Data1.height-10));
  but_Data1.draw(panelImage);
  but_Data2.draw(panelImage);
  but_Data3.draw(panelImage);
  but_Data4.draw(panelImage);
  but_Data5.draw(panelImage);
  but_Quit.draw(panelImage);

  float neuronStrength, modulationLevel, val;
  float k = 3;

  char key;

  initDataSet1(V2, attentionPoint, k);

  do {
    outputImage.clear();
    for(int theta_index = 0; theta_index < THETA_LEVELS; theta_index++) {
      for(int x = 0; x<MAP_SIZE; x++) {
        for(int y = 0; y<MAP_SIZE; y++) {
          float theta = float(theta_index)/float(THETA_LEVELS)*2.0F*PI;

          modulationLevel = M(gaussianKernel, theta, Point2D<int>(x,y), attentionPoint);
          neuronStrength=V2[theta_index].getVal(x,y);
          val = modulationLevel * neuronStrength * k;

          //Draw the original V2 State
          if(neuronStrength > 0) {
            drawHalfLine(outputImage, Point2D<int>(x*OUT_SCALE_FACT, y*OUT_SCALE_FACT), -theta, OUT_SCALE_FACT*3/4, PixRGB<byte>(75,75,75));
          }

          //Draw the attentional Modulation Vectors
          if(val > 0) {

            PixRGB<byte> curr = outputImage.getVal(x*OUT_SCALE_FACT,y*OUT_SCALE_FACT);




              drawHalfLine(outputImage, Point2D<int>(x*OUT_SCALE_FACT, y*OUT_SCALE_FACT), -(theta-PI/2.0f), val*10,
                         PixRGB<byte>(curr + PixRGB<byte>(255*val*BRIGHTNESS,255*val*BRIGHTNESS, 0)) );

            drawHalfLine(outputImage, Point2D<int>(x*OUT_SCALE_FACT, y*OUT_SCALE_FACT), theta, OUT_SCALE_FACT/2,
                         PixRGB<byte>(curr + PixRGB<byte>(255*val*BRIGHTNESS,255*val*BRIGHTNESS, 0)) );

             drawHalfLine(outputImage, Point2D<int>(x*OUT_SCALE_FACT, y*OUT_SCALE_FACT), -theta, OUT_SCALE_FACT/2,
                          PixRGB<byte>(curr + PixRGB<byte>(255*val*BRIGHTNESS,255*val*BRIGHTNESS, 0)) );
          }


        }
      }
    }
    /*

    Image<float> temp =  Image<float>(Dims(OUTPUT_RES, OUTPUT_RES), ZEROS);
    for(int i=0; i<MAP_SIZE; i++) {
      for(int j=0; j<MAP_SIZE; j++) {
        float val =  M(gaussianKernel, 0, Point2D<int>(MAP_SIZE/2,MAP_SIZE/2), Point2D<int>(i,j));
        temp.setVal(i*OUT_SCALE_FACT, j*OUT_SCALE_FACT, val);
      }
    }
    flipVertical(temp);
    window2->drawImage(temp,0,0);
    */



    //Draw the attention point as a red cross
    drawCross(outputImage, attentionPoint*OUT_SCALE_FACT, PixRGB<byte>(255,0,0));

    /*drawHalfLine(outputImage, attentionPoint*OUT_SCALE_FACT, PI, 30,
      PixRGB<byte>(PixRGB<byte>(255,255, 0)) );*/

    char buffer[20];
    sprintf(buffer,"(%d, %d)", attentionPoint.i, attentionPoint.j);

    //Because the image coordinates have y=0 at the top of the screen, the output is flipped to give you a more sane image
    flipVertical(outputImage);

    //Write the attention point coordinates
    writeText(outputImage, Point2D<int>(attentionPoint.i*OUT_SCALE_FACT+5, outputImage.getHeight() - attentionPoint.j*OUT_SCALE_FACT+15), buffer, PixRGB<byte>(255,0,0), PixRGB<byte>(0,0,0), SimpleFont::FIXED(6), false);

    //Draw the border ownership model, and the interface panel into the display window
    window1->drawImage(outputImage,0,0);
    window1->drawImage(panelImage,outputImage.getWidth(),0);


    key = window1->getLastKeyPress();
    //These key presses may be OS dependent. If you want keyboard support, just put a print statement here,
    //print out the keys as they are pressed, and map them to the correct actions.
    switch(key) {
    case -122:
      attentionPoint.j++;
      break;
    case -123:
      attentionPoint.j--;
      break;
    case -125:
      attentionPoint.i--;
      break;
    case -124:
      attentionPoint.i++;
      break;
    case 26:
      initDataSet1(V2, attentionPoint, k);
      break;
    case 27:
      initDataSet2(V2, attentionPoint, k);
      break;
    case 28:
      initDataSet3(V2, attentionPoint, k);
      break;
    case 29:
      initDataSet4(V2, attentionPoint, k);
      break;
    case 31:
      initDataSet5(V2, attentionPoint, k);
      break;
    }

    Point2D<int> p;
    p = window1->getLastMouseClick();
    if (p.isValid()) {
      //Check for button hits
      Point2D<int> panelPoint = Point2D<int>(p.i-OUTPUT_RES, p.j);
      if(but_Data1.inBounds(panelPoint)) {
        initDataSet1(V2, attentionPoint, k);
      }
      if(but_Data2.inBounds(panelPoint)) {
        initDataSet2(V2, attentionPoint, k);
      }
      if(but_Data3.inBounds(panelPoint)) {
        initDataSet3(V2, attentionPoint, k);
      }
      if(but_Data4.inBounds(panelPoint)) {
        initDataSet4(V2, attentionPoint, k);
      }
      if(but_Data5.inBounds(panelPoint)) {
        initDataSet5(V2, attentionPoint, k);
      }
      if(but_Quit.inBounds(panelPoint)) {
        LINFO("Qutting... Thanks!");
        exit(0);
      }
      //Otherwise, place the attention point on the mouse click
      else if(p.i < OUTPUT_RES) {
        p/=OUT_SCALE_FACT;

        p.j = MAP_SIZE-p.j;
        attentionPoint = p;
      }
    }
  } while(key != 20);
}

//Given a certain point i in an orientation level theta in V2, and an attention point,
//find the top-down attentional modulation level of that point.
//This simulates the synaptic weightings between V2 and an attentional layer, whether that be top-down or bottom-up
float M(Image<float> WeightKernel, float theta, Point2D<int> i, Point2D<int> att) {

  if(i.distance(att) > WeightKernel.getWidth()/2 || i.distance(att) == 0)
    return 0.0F;

  float theta_prime = beta(i,att);

  if(theta_prime == theta || (theta_prime >= theta + PI -.01 && theta_prime <= theta+PI+.01) || (theta_prime >= theta - PI -.01 && theta_prime <= theta-PI+.01))
    return 0.0f;

  if((theta <= PI && theta_prime >= theta && theta_prime <= theta+PI) ||
     (theta > PI && (theta_prime >= theta || theta_prime < theta-PI)))
    return 0;
  else {
    return WeightKernel.getVal(WeightKernel.getWidth()/2 + abs(i.i - att.i),WeightKernel.getHeight()/2 + abs(i.j - att.j));
  }
}


//Returns the angle between i and j. The angle returned is in radians, and ranges from 0-2PI
float beta(Point2D<int> i, Point2D<int> j) {

  if(j.j-i.j>=0)
    return atan2(j.j - i.j, j.i - i.i);
  else
    return 2.0f*PI + atan2(j.j - i.j, j.i - i.i);
}


//Flip an image on about it's x axis
template <class T>
void flipVertical(Image <T> &Image) {
  T temp;

  for(int x = 0; x<Image.getWidth(); x++) {
    for(int y=0; y<Image.getHeight()/2; y++) {
      temp = Image.getVal(x,Image.getHeight() - y - 1);
      Image.setVal(x,Image.getHeight() - y - 1, Image.getVal(x,y));
      Image.setVal(x,y,temp);
    }
  }
}


//Initialize V2 with data set 1: A single vertical line with ambiguous ownership.
//The attention point is initially to the right of the line, but the viewer should manually move it around to watch the model pick
void initDataSet1(vector< Image<float> > &V2, Point2D<int> &attentionPoint, float &k) {
  int theta_index = THETA_LEVELS/4;
  int theta_index2 = THETA_LEVELS*3/4;
  Dims d = V2[0].getDims();

  k = 7.0f;

  //Clear out V2
  for(int i=0; i<THETA_LEVELS; i++)
   V2[i].clear();

  //Draw a 1 pixel line through the middle of the PI/2 layer of V2
  //This effectively simulates the activation of a right owned vertical line
  drawLine(V2[theta_index], Point2D<int>(d.w()/2,0), Point2D<int>(d.w()/2,d.h()), 1.0F);

  //Draw a 1 pixel line through the middle of the 3PI/2 layer of V2
  //This effectively simulates the activation of a left owned vertical line
  drawLine(V2[theta_index2], Point2D<int>(d.w()/2,0), Point2D<int>(d.w()/2,d.h()), 1.0F);


  //Put the attention point 5% to the right of the line
  attentionPoint = Point2D<int>(int(d.w()/2+d.w()*.05), d.h()/2);
}

//Initialize V2 with data set 2: A horizontal vertical line with ambiguous ownership.
//The attention point is initially to the above the line, but the viewer should manually move it around to watch the model pick
void initDataSet2(vector< Image<float> > &V2, Point2D<int> &attentionPoint, float &k) {
  int theta_down_right = THETA_LEVELS/8;
  int theta_up_left = THETA_LEVELS*5/8;
  Dims d = V2[0].getDims();

  k = 7.0f;

  //Clear out V2
  for(int i=0; i<THETA_LEVELS; i++)
   V2[i].clear();

  //Draw a 1 pixel line through the middle of the PI layer of V2
  //This effectively simulates the activation of an upwards owned horizontal line
  drawLine(V2[theta_up_left], Point2D<int>(0,0), Point2D<int>(d.w()-1,d.h()-1), 1.0F);

  //Draw a 1 pixel line through the middle of the 0 layer of V2
  //This effectively simulates the activation of a downwards owned horizontal line
   drawLine(V2[theta_down_right], Point2D<int>(0,0), Point2D<int>(d.w()-1,d.h()-1), 1.0F);


  //Put the attention point 5% to the right of the line
  attentionPoint = Point2D<int>(d.w()/2, int(d.h()/2+d.h()*.05));
}


//Initialize V2 with data set 3: An ambigously owned square in the middle of the image
//The attention point is initially to the right of the line, but the viewer should manually move it around to watch the model pick
void initDataSet3(vector< Image<float> > &V2, Point2D<int> &attentionPoint, float &k) {

  Dims d = V2[0].getDims();

  int theta_down = 0;
  int theta_right = THETA_LEVELS/4;
  int theta_up = THETA_LEVELS/2;
  int theta_left = THETA_LEVELS*3/4;

  Point2D<int> topRight = Point2D<int>(d.w()*3/4, d.h()*3/4);
  Point2D<int> topLeft = Point2D<int>(d.w()/4, d.h()*3/4);
  Point2D<int> bottomRight = Point2D<int>(d.w()*3/4, d.h()/4);
  Point2D<int> bottomLeft = Point2D<int>(d.w()/4, d.h()/4);

  k = 7.0f;

 //Clear out V2
 for(int i=0; i<THETA_LEVELS; i++)
   V2[i].clear();

  //Draw the right line with bown to the right
  drawLine(V2[theta_right], bottomRight, topRight, 1.0F);
  //Draw the right line with bown to the left
  drawLine(V2[theta_left], bottomRight, topRight, 1.0F);

  //Draw the top line with bown up
  drawLine(V2[theta_up],topLeft, topRight, 1.0F);
  //Draw the top line with bown down
  drawLine(V2[theta_down],topLeft, topRight, 1.0F);

  //Draw the left line with bown right
  drawLine(V2[theta_right],topLeft, bottomLeft, 1.0F);
  //Draw the left line with bown left
  drawLine(V2[theta_left],topLeft, bottomLeft, 1.0F);

  //Draw the bottom line with bown up
  drawLine(V2[theta_up],bottomLeft, bottomRight, 1.0F);
  //Draw the bottom line with bown down
  drawLine(V2[theta_down],bottomLeft, bottomRight, 1.0F);

  attentionPoint = Point2D<int>(d.w()/2, d.h()/2);
}


//Initialize V2 with data set 4: A set of 5 ambiguously owned squares, sharing borders
void initDataSet4(vector< Image<float> > &V2, Point2D<int> &attentionPoint, float &k) {
  int theta_down = 0;
  int theta_right = THETA_LEVELS/4;
  int theta_up = THETA_LEVELS/2;
  int theta_left = THETA_LEVELS*3/4;

  Dims d = V2[0].getDims();

  k = 2.5f;

  //Clear out V2
  for(int i=0; i<THETA_LEVELS; i++)
    V2[i].clear();

  drawLine(V2[theta_up],   Point2D<int>(d.w()/8,d.h()*3/8), Point2D<int>(d.w()*7/8,d.h()*3/8), 1.0F);
  drawLine(V2[theta_down], Point2D<int>(d.w()/8,d.h()*3/8), Point2D<int>(d.w()*7/8,d.h()*3/8), 1.0F);

  drawLine(V2[theta_up],   Point2D<int>(d.w()/8,d.h()*5/8), Point2D<int>(d.w()*7/8,d.h()*5/8), 1.0F);
  drawLine(V2[theta_down], Point2D<int>(d.w()/8,d.h()*5/8), Point2D<int>(d.w()*7/8,d.h()*5/8), 1.0F);

  drawLine(V2[theta_left],  Point2D<int>(d.w()/8,d.h()*3/8), Point2D<int>(d.w()/8,d.h()*5/8), 1.0F);
  drawLine(V2[theta_right], Point2D<int>(d.w()/8,d.h()*3/8), Point2D<int>(d.w()/8,d.h()*5/8), 1.0F);

  drawLine(V2[theta_left],  Point2D<int>(d.w()*7/8,d.h()*3/8), Point2D<int>(d.w()*7/8,d.h()*5/8), 1.0F);
  drawLine(V2[theta_right], Point2D<int>(d.w()*7/8,d.h()*3/8), Point2D<int>(d.w()*7/8,d.h()*5/8), 1.0F);



  drawLine(V2[theta_left],   Point2D<int>(d.w()*3/8, d.h()/8), Point2D<int>(d.w()*3/8,d.h()*7/8), 1.0F);
  drawLine(V2[theta_right],  Point2D<int>(d.w()*3/8, d.h()/8), Point2D<int>(d.w()*3/8,d.h()*7/8), 1.0F);

  drawLine(V2[theta_left],   Point2D<int>(d.w()*5/8, d.h()/8), Point2D<int>(d.w()*5/8,d.h()*7/8), 1.0F);
  drawLine(V2[theta_right],  Point2D<int>(d.w()*5/8, d.h()/8), Point2D<int>(d.w()*5/8,d.h()*7/8), 1.0F);

  drawLine(V2[theta_up],   Point2D<int>(d.w()*3/8,d.h()/8), Point2D<int>(d.w()*5/8,d.h()/8), 1.0F);
  drawLine(V2[theta_down], Point2D<int>(d.w()*3/8,d.h()/8), Point2D<int>(d.w()*5/8,d.h()/8), 1.0F);

  drawLine(V2[theta_up],   Point2D<int>(d.w()*3/8,d.h()*7/8), Point2D<int>(d.w()*5/8,d.h()*7/8), 1.0F);
  drawLine(V2[theta_down], Point2D<int>(d.w()*3/8,d.h()*7/8), Point2D<int>(d.w()*5/8,d.h()*7/8), 1.0F);

  //Put the attention point 5% to the right of the line
  attentionPoint = Point2D<int>(d.w()/2, int(d.h()/2+d.h()*.05));
}



//Initialize V2 with data set 5: A rough imitation of Rubins Vase
void initDataSet5(vector< Image<float> > &V2, Point2D<int> &attentionPoint, float &k) {

  Dims d = V2[0].getDims();

  int x1  = d.w()/12;
  int x4  = d.w()/3;
  int x8  = d.w()*8/12;
  int x11 = d.w()*11/12;

  k = 4.0f;

  //Clear out V2
  for(int i=0; i<THETA_LEVELS; i++)
    V2[i].clear();

  drawLine(V2[THETA_LEVELS*3/8], Point2D<int>(x1,x11), Point2D<int>(x4,x8), 1.0F);
  drawLine(V2[THETA_LEVELS*7/8], Point2D<int>(x1,x11), Point2D<int>(x4,x8), 1.0F);

  drawLine(V2[THETA_LEVELS/4],   Point2D<int>(x4,x8), Point2D<int>(x4,x4), 1.0F);
  drawLine(V2[THETA_LEVELS*3/4], Point2D<int>(x4,x8), Point2D<int>(x4,x4), 1.0F);

  drawLine(V2[THETA_LEVELS/8],   Point2D<int>(x4,x4), Point2D<int>(x1,x1), 1.0F);
  drawLine(V2[THETA_LEVELS*5/8], Point2D<int>(x4,x4), Point2D<int>(x1,x1), 1.0F);

  drawLine(V2[THETA_LEVELS/8], Point2D<int>(d.w()-x1,x11), Point2D<int>(d.w()-x4,x8), 1.0F);
  drawLine(V2[THETA_LEVELS*5/8], Point2D<int>(d.w()-x1,x11), Point2D<int>(d.w()-x4,x8), 1.0F);

  drawLine(V2[THETA_LEVELS/4],   Point2D<int>(d.w()-x4,x8), Point2D<int>(d.w()-x4,x4), 1.0F);
  drawLine(V2[THETA_LEVELS*3/4], Point2D<int>(d.w()-x4,x8), Point2D<int>(d.w()-x4,x4), 1.0F);

  drawLine(V2[THETA_LEVELS*3/8],   Point2D<int>(d.w()-x4,x4), Point2D<int>(d.w()-x1,x1), 1.0F);
  drawLine(V2[THETA_LEVELS*7/8], Point2D<int>(d.w()-x4,x4), Point2D<int>(d.w()-x1,x1), 1.0F);
}


template <class T>
void drawHalfLine(Image<T>& dst,
    const Point2D<int>& pos, float ori, float len, const T col,
    const int rad = 1)
{

GVX_TRACE(__PRETTY_FUNCTION__);
  ASSERT(dst.initialized());

  int x1 = int(cos(ori)*len);
  int y1 = int(sin(ori)*len);

  Point2D<int> p1 = pos;
  Point2D<int> p2(pos.i+x1, pos.j-y1);

  drawLine(dst, p1, p2, col, rad);

}

