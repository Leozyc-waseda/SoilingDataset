/*! @file BeoSub/test-canny.C [put description here] */

/*******************************************************************************
* --------------------------------------------
*(c) 2001 University of South Florida, Tampa
* Use, or copying without permission prohibited.
* PERMISSION TO USE
* In transmitting this software, permission to use for research and
* educational purposes is hereby granted.  This software may be copied for
* archival and backup purposes only.  This software may not be transmitted
* to a third party without prior permission of the copyright holder. This
* permission may be granted only by Mike Heath or Prof. Sudeep Sarkar of
* University of South Florida (sarkar@csee.usf.edu). Acknowledgment as
* appropriate is respectfully requested.
*
*  Heath, M., Sarkar, S., Sanocki, T., and Bowyer, K. Comparison of edge
*    detectors: a methodology and initial study, Computer Vision and Image
*    Understanding 69 (1), 38-54, January 1998.
*  Heath, M., Sarkar, S., Sanocki, T. and Bowyer, K.W. A Robust Visual
*    Method for Assessing the Relative Performance of Edge Detection
*    Algorithms, IEEE Transactions on Pattern Analysis and Machine
*    Intelligence 19 (12),  1338-1359, December 1997.
*  ------------------------------------------------------
*
* PROGRAM: canny_edge
* PURPOSE: This program implements a "Canny" edge detector. The processing
* steps are as follows:
*
*   1) Convolve the image with a separable gaussian filter.
*   2) Take the dx and dy the first derivatives using [-1,0,1] and [1,0,-1]'.
*   3) Compute the magnitude: sqrt(dx*dx+dy*dy).
*   4) Perform non-maximal suppression.
*   5) Perform hysteresis.
*
* The user must input three parameters. These are as follows:
*
*   sigma = The standard deviation of the gaussian smoothing filter.
*   tlow  = Specifies the low value to use in hysteresis. This is a
*           fraction (0-1) of the computed high threshold edge strength value.
*   thigh = Specifies the high value to use in hysteresis. This fraction (0-1)
*           specifies the percentage point in a histogram of the gradient of
*           the magnitude. Magnitude values of zero are not counted in the
*           histogram.
*
* NAME: Mike Heath
*       Computer Vision Laboratory
*       University of South Floeida
*       heath@csee.usf.edu
*
* DATE: 2/15/96
*
* Modified: 5/17/96 - To write out a floating point RAW headerless file of
*                     the edge gradient "up the edge" where the angle is
*                     defined in radians counterclockwise from the x direction.
*                     (Mike Heath)
*******************************************************************************/

//
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-canny.C $
// $Id: test-canny.C 14857 2011-07-26 01:14:21Z siagian $
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
//


/**************************TODO*************************
    -NOTE: switching to std vecotrs BREAKS powell, even if changed throughout. Note sure why

    -MAKE a tester execuatable to spool camera values (similar to those displayed at the start of test-multigrab) continually, checking white balance, exposure and gain in order to manually clibrate the cameras at wet test time. Auto set is not working well. NOTE that every color setting is currently rcognizing white! this is bad! something wrong with the V range? <-seems fixable using a mix of changing V and hardware stuff

    -SUGGESTION: instead of returning final recognized position data using a struct, inseat store data in shape class and return using reference. (allows shapes with different dimensions to return useful data, unlike current standard 5 dims)
*******************************************************/

#include "BeoSub/CannyModel.H"
#include "BeoSub/hysteresis.H"
#include "Component/ModelManager.H"
#include "Devices/FrameGrabberFactory.H"
#include "GUI/XWindow.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/MorphOps.H"     // for erodeImg(), closeImg()
#include "Image/Transforms.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H" // for squareOf()
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "VFAT/segmentImageTrackMC.H"
#include "rutz/shared_ptr.h"

#include <algorithm> // for std::max()
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

//canny
#define BOOSTBLURFACTOR 90.0
//powell
#define TOL 2.0e-4
#define ITMAX 200
//brent...
#define ITMAXB 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
#define SIGN(a,b) ((b)>=0.0?fabs(a):-fabs(a))
//mnbrak
#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20

//CAMERA STUFF


#define FREE_ARG char*
//END CAMERA STUFF

int canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, const char* fname);
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
        short int **smoothedim);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void derrivative_x_y(short int *smoothedim, int rows, int cols,
        short int **delta_x, short int **delta_y);
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
        short int **magnitude);
void radian_direction(short int *delta_x, short int *delta_y, int rows,
    int cols, float **dir_radians, int xdirtag, int ydirtag);
void radian_direction(short int *delta_x, short int *delta_y, int rows,
    int cols, float **dir_radians, int xdirtag, int ydirtag);
double angle_radians(double x, double y);
void grabImage(Image<PixRGB<byte> >* image);

Image<float> distMap;
rutz::shared_ptr<XWindow> xwin, win;
const bool debugmode = true;

//Moved to global

bool fromFile = true;//boolean to check if input is from file or from camera

const char *infilename = NULL;  /* Name of the input image */
const char *dirfilename = NULL; /* Name of the output gradient direction image */
const char *shapeArg = NULL;       /* Shape for recognition*/
const char *colorArg = NULL;       /* Color for tracking*/
char outfilename[128];    /* Name of the output "edge" image */
char composedfname[128];  /* Name of the output "direction" image */
unsigned char *edge;      /* The output edge image */
float sigma,              /* Standard deviation of the gaussian kernel. */
    tlow,               /* Fraction of the high threshold in hysteresis. */
    thigh;              /* High hysteresis threshold control. The actual
                           threshold is the (100 * thigh) percentage point
                           in the histogram of the magnitude of the
                           gradient image that passes non-maximal
                           suppression. */

int imgW, imgH, ww, hh;
Image< PixRGB<byte> > colorImg;
Image<byte> grayImg;

Image< PixRGB<byte> > display;
Image<byte> temp;
Image<PixRGB<byte> > Aux;
Image<PixH2SV2<float> > H2SVimage;

// instantiate a model manager (for camera input):
ModelManager manager("Canny Tester");
// Instantiate our various ModelComponents:
nub::soft_ref<FrameIstream>
gb(makeIEEE1394grabber(manager, "cannycam", "cc"));
/// all this should not be in global scope, but in main()!

segmentImageTrackMC<float,unsigned int,4> *segmenter;

//! Mean color to track
std::vector<float> color(4,0.0F);

bool runCanny(bool fromFile, bool useColor, const char* shapeArg); //FIX: change later so this returns struct with found info (position, etc)

/************************************************************
Optimizer function Prototypes (pt2)
************************************************************/

//back to being ripped from a book

//Note: powell and linmin now use a ShapeModel optimizee class. brent and mnbrak use func ptr to f1dim, which has copy of the optimizee
bool powell(double p[], double **xi, int n, double ftol,
            int *iter, double *fret, rutz::shared_ptr<ShapeModel>& optimizee);
int ncom;
double *pcom,*xicom;

double brent(double ax, double bx, double cx,
             double (*f)(double, rutz::shared_ptr<ShapeModel>&), double tol,
             double *xmin, rutz::shared_ptr<ShapeModel>& shape);
void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb,
            double *fc, double (*func)(double, rutz::shared_ptr<ShapeModel>&),
            rutz::shared_ptr<ShapeModel>& shape);
double f1dim(double x, rutz::shared_ptr<ShapeModel>& shape);
void linmin(double p[], double xi[], int n, double *fret,
            rutz::shared_ptr<ShapeModel>& optimizee);


///////////////////////FROM NRUTIL/////////////////////////


double *nrVector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
        double *v;

        v=(double *)malloc((size_t) ((nh-nl+2)*sizeof(double)));
        if (!v) printf("allocation failure in nrVector()");
        return v-nl+1;
}

void free_nrVector(double *v, long nl, long nh)
/* free a double vector allocated with nrVector() */
{
        free((FREE_ARG) (v+nl-1));
}

//////////////////////////MAIN/////////////////////////////

int main(int argc, char *argv[])
{
  //  MYLOGVERB = 0; //sets extern value to suppress unwanted output from other modules

  //set math details. For now, harcoded
  sigma = .5;
  tlow = .5;
  thigh = .5;

  bool useColor = true;

  manager.addSubComponent(gb);


  // set the camera number (in IEEE1394 lingo, this is the
  // "subchannel" number):
  gb->setModelParamVal("FrameGrabberSubChan", 0);



  /****************************************************************************
   * Get the command line arguments.
   ****************************************************************************/
  if(argc < 3){
    fprintf(stderr,"\n<USAGE> %s image shape color \n",argv[0]);
    fprintf(stderr,"\n      image:      An image to process. Must be in PGM format.\n");
    fprintf(stderr,"                  Type 'none' for camera input.\n");
    fprintf(stderr,"      shape:      Shape on which to run recognition\n");
    fprintf(stderr,"                  Candidates: Rectangle, Square, Circle, Octagon.\n");
    fprintf(stderr,"      color:       Color to track\n");
    fprintf(stderr,"                  Candidates: Blue, Yellow, none (for no color tracking).\n");
    exit(1);
  }
  infilename = argv[1];
  shapeArg = argv[2];
  colorArg = argv[3];

  printf("READ: 1: %s 2: %s 3: %s\n", infilename, shapeArg, colorArg);

  if(!strcmp(infilename, "none")){
    fromFile = false;
    infilename = "../../bin/Camera.pgm";//hacky~ should let user say where they want file placed
  }

  //what's this?
  if(argc == 6) dirfilename = infilename;
  else dirfilename = NULL;


  /***************************************************************************
   * Perform color exclusion. Using grayImg for now, should likely use separate image later to preserve step output.
   *NOTE: a lot of code is repeated in file and camera input, because up in-if/else declaration of segmenter
   **************************************************************************/


  //COLORTRACKING DECS: note that much of these may not need to be in this file
  //Colors MUST use H2SV2 pixel values! use test-sampleH2SV2 to sample needed values! --Z

  //0 = H1 (0-1), 1=H2 (0-1), 2=S (0-1), 3=V (0-1)
  if(!strcmp(colorArg, "Red")){
    //RED
    color[0] = 0.70000F; color[1] = 0.94400F; color[2] = 0.67200F; color[3] = 0.96900F;
  }

  else if(!strcmp(colorArg, "Green")){
    //Green
    color[0] = 0.53000F; color[1] = 0.36000F; color[2] = 0.44600F; color[3] = 0.92000F;
  }

  else if(!strcmp(colorArg, "Orange")){
    //"ORANGE"
    color[0] = 0.80000F; color[1] = 0.80000F; color[2] = 0.57600F; color[3] = 0.96800F;
  }
  else if(!strcmp(colorArg, "Blue")){
    //BLUE
    color[0] = 0.27500F; color[1] = 0.52500F; color[2] = 0.44500F; color[3] = 0.97200F;
  }
  else if(!strcmp(colorArg, "Yellow")){
    //YELLOW
    color[0] = 0.90000F; color[1] = 0.62000F; color[2] = 0.54000F; color[3] = 0.97000F;
  }
  else{
    useColor = false;
  }

   //! +/- tollerance value on mean for track
   std::vector<float> std(4,0.0F);
   //NOTE that the saturation tolerance is important (if it gos any higher thn this, it will nearly always recognize white!)
   std[0] = 0.30000; std[1] = 0.30000; std[2] = 0.44500; std[3] = 0.45000;

   //! normalizer over color values (highest value possible)
   std::vector<float> norm(4,0.0F);
   norm[0] = 1.0F; norm[1] = 1.0F; norm[2] = 1.0F; norm[3] = 1.0F;

   //! how many standard deviations out to adapt, higher means less bias
   //The lower these are, the more strict recognition will be in subsequent frames
   //TESTED AND PROVED do NOT change unless you're SURE
   std::vector<float> adapt(4,0.0F);
   //adapt[0] = 3.5F; adapt[1] = 3.5F; adapt[2] = 3.5F; adapt[3] = 3.5F;
   adapt[0] = 5.0F; adapt[1] = 5.0F; adapt[2] = 5.0F; adapt[3] = 5.0F;

   //! highest value for color adaptation possible (hard boundry)
   std::vector<float> upperBound(4,0.0F);
   upperBound[0] = color[0] + 0.45F; upperBound[1] = color[1] + 0.45F; upperBound[2] = color[2] + 0.55F; upperBound[3] = color[3] + 0.55F;

   //! lowest value for color adaptation possible (hard boundry)
   std::vector<float> lowerBound(4,0.0F);
   lowerBound[0] = color[0] - 0.45F; lowerBound[1] = color[1] - 0.45F; lowerBound[2] = color[2] - 0.55F; lowerBound[3] = color[3] - 0.55F;
   //END DECS

   //set up image details
   if(fromFile){
     colorImg = Raster::ReadRGB(infilename);
     imgW = colorImg.getWidth();
     imgH = colorImg.getHeight();
   }
   else{
     /**************************************************
       CAMERA INPUT
        -Code ripped from test-grab. Forces to IEEE1394. has short stream for buffering and reads in pic from camera, then starts powell minimization, which displays in new window.
     **************************************************/

     imgH = gb->getHeight();
     imgW = gb->getWidth();

 }

   segmenter = new segmentImageTrackMC<float,unsigned int,4> (imgW*imgH);
   //color tracking stuff
   segmenter->SITsetTrackColor(&color,&std,&norm,&adapt,&upperBound,&lowerBound);

   ww = imgW/4;
   hh = imgH/4;

   segmenter->SITsetFrame(&ww,&hh);

   // Set display colors for output of tracking. Strictly aesthetic //TEST
   //And definitely not needed, although for some reason it's required --Z
   segmenter->SITsetCircleColor(0,255,0);
   segmenter->SITsetBoxColor(255,255,0,0,255,255);
   segmenter->SITsetUseSmoothing(true,10);

   //method provided by Nathan to turn off adaptation (useful for dynamic environments)
   segmenter->SITtoggleColorAdaptation(false);
   //enable single-frame tracking
   segmenter->SITtoggleCandidateBandPass(false);

   Image<PixRGB<byte> > Aux;
   Aux.resize(100,450,true);
   Image<byte> temp;
   Image< PixRGB<byte> > display = colorImg;

   if(fromFile){   //for input from file

     if(useColor){
       segmenter->SITtrackImage(colorImg,&display,&Aux,true);
       temp = segmenter->SITreturnCandidateImage();//get image we'll be using for edge detection

       grayImg = luminance(temp);
     }
     else{
       grayImg = luminance(colorImg);
     }
   }
   else{
     // get ready for main loop:
     win.reset( new XWindow(gb->peekDims(), -1, -1, "test-canny window") );
     //color tracking stuffages
     segmenter->SITsetTrackColor(&color,&std,&norm,&adapt,&upperBound,&lowerBound);
     ww = imgW/4;
     hh = imgH/4;
     segmenter->SITsetFrame(&ww,&hh);
     // Set display colors for output of tracking. Strictly asthetic //TEST
     //And definitely not needed, although for soe reason it's required --Z
     segmenter->SITsetCircleColor(0,255,0);
     segmenter->SITsetBoxColor(255,255,0,0,255,255);
     segmenter->SITsetUseSmoothing(true,10);

     Aux.resize(100,450,true);

   }

   if(fromFile){
     runCanny(fromFile, useColor, shapeArg);
   }
   else{

     // let's get all our ModelComponent instances started:
     manager.start();

     // get the frame grabber to start streaming:
     gb->startStream();

     colorImg = gb->readRGB();

     xwin.reset(new XWindow(colorImg.getDims()));
     xwin->setPosition(colorImg.getWidth(), 0);
     while(1){
       runCanny(fromFile, useColor, shapeArg);
     }

    manager.stop();
   }
}

bool runCanny(bool fromFile, bool useColor, const char* shapeArg){

  if(!fromFile){
    int i;
    for(i = 0; i < 30; i++) {//give camera holder time to aim

      colorImg = gb->readRGB();
      win->drawImage(colorImg);
    }

    //Get candidate image representing blobs of desired color
    H2SVimage = colorImg;//changed since transfer to H2SV
    display = colorImg;
    segmenter->SITtrackImageAny(H2SVimage,&display,&Aux,true);
    temp = segmenter->SITreturnCandidateImage();//get image we'll be using for edge detection

    //set up structuring elements (disk)
    Image<byte> se1(5, 5, ZEROS);
    Point2D<int> t1(2,2);
    drawDisk(se1, t1, 2, static_cast<byte>(255));

    Image<byte> se2(9, 9, ZEROS);
    Point2D<int> t2(4,4);
    drawDisk(se2, t2, 3, static_cast<byte>(255));

    //Erode image to prevent noise NOTE: isn't working well, erodes too much, probably because temp is so lowrez
    //temp = erodeImg(temp, se1);

    //Close candidate image to prevent holes
    temp = closeImg(temp, se2);

    if(useColor){
      grayImg = (Image<byte>)(quickInterpolate(temp,4));//note that this is very low rez
    }
    else{
      grayImg = luminance(colorImg);
    }
    //END CAMERA
  }
   /****************************************************************************
    * Perform the edge detection.
    ****************************************************************************/
  /*
   if(dirfilename != NULL){
     if(!fromFile){
       sprintf(composedfname, "%s_s_%3.2f_l_%3.2f_h_%3.2f.fim", infilename,
               sigma, tlow, thigh);
     }
     else{
       sprintf(composedfname, "%s_s_%3.2f_l_%3.2f_h_%3.2f.fim", "Camera",
               sigma, tlow, thigh);
     }
     dirfilename = composedfname;
   }
  */
   canny(grayImg.getArrayPtr(), grayImg.getHeight(), grayImg.getWidth(), sigma, tlow, thigh, &edge, dirfilename);


   /****************************************************************************
    * Write out the edge image to a file.
    *****************************************************************************/
   /*
   if(!fromFile){
     sprintf(outfilename, "%s_s_%3.2f_l_%3.2f_h_%3.2f.pgm", infilename, sigma, tlow, thigh);
   }
   else{
     sprintf(outfilename, "%s_s_%3.2f_l_%3.2f_h_%3.2f.pgm", "Camera", sigma, tlow, thigh);
   }
   */
   Image<unsigned char> cannyCharImg(edge, grayImg.getWidth(), grayImg.getHeight());

   Image<float> cannyImg = cannyCharImg;

   Raster::WriteGray(cannyImg, outfilename, RASFMT_PNM);//edge image

   xwin->drawImage((Image< PixRGB<byte> >)grayImg);

   Point2D<int> cent = centroid(cannyCharImg);
   distMap = chamfer34(cannyImg, 5000.0f);
   //END EDGE DETECTION

   //PERFORM SHAPE DETECTION
   rutz::shared_ptr<ShapeModel> shape;
   double* p = NULL;

   int iter = 0;
   double fret = 0.0;
   bool shapeFound = false;

   //NOTE: following bracket set is to define scope outside which the shape model will be destroyed
   {
     /***********************************************
      *Set values for the model and declare it here
        Common Values:
           x_center: initial central x position of the model
           y_center: initial central y position of the model
              -in the future, try multiple arrangements of these,
                    perhaps in some meta-omptimizer, since noisy
                    images are sensitive to these values
           height/width/etc: initial dimensions of the model.
               Be careful to note if ratio is to be fixed or free!
           alpha: initial angular orientation of the model
     **********************************************/

     if(!strcmp(shapeArg, "Rectangle")){
       //rectangle
       //NOTE: the order of dimensions is changed. This is a test. May need to change thm back

       p = (double*)calloc(6, sizeof(double));
       p[1] = (double)(cent.i); //Xcenter
       p[2] = (double)(cent.j); //Ycenter
       p[4] = 170.0f; // Width
       p[5] = 140.0f; // Height
       p[3] = 0.0; //alpha

       shape.reset(new RectangleShape(120.0, p, true));
     }
     else if(!strcmp(shapeArg, "Square")){
       //try it with a square

       p = (double*)calloc(5, sizeof(double));
       p[1] = (double)(cent.i); //Xcenter
       p[2] = (double)(cent.j); //Ycenter
       p[3] = 100.0f; // Height
       p[4] = 0.0; // alpha

       shape.reset(new SquareShape(100.0, p, true));
     }
     else if(!strcmp(shapeArg, "Octagon")){
       //try it with an octagon

       p = (double*)calloc(5, sizeof(double));
       p[1] = (double)(cent.i); //Xcenter
       p[2] = (double)(cent.j); //Ycenter
       p[3] = 80.0f; // Height
       p[4] = 0.0; // alpha

       shape.reset(new OctagonShape(80.0, p, true));
     }
     //    else if(!strcmp(shapeArg, "Parallel")){
     //        //parallel lines (SUCKS for now)
     //        p[1] = (double)(cent.i); //Xcenter
     //        p[2] = (double)(cent.j); //Ycenter
     //        p[3] = 50.0f; // Width
     //        p[4] = 160.0f; // Height
     //        p[5] = 0.0; //alpha

     //        shape = new ParallelShape(40.0, true);
     //      }
     else if(!strcmp(shapeArg, "Circle")){
       //circle
       p = (double*)calloc(4, sizeof(double));
       p[1] = (double)(cent.i); //Xcenter
       p[2] = (double)(cent.j); //Ycenter
       p[3] = 50.0f; // Radius


       shape.reset(new CircleShape(40.0, p, true));
     }
     else{
       printf("Cannot run shape recognition without a shape to recognize! Returning...\n");
       shape.reset(new CircleShape(9999.0, p, false));
       return(false);
     }

     double** xi = (double**)calloc(shape->getNumDims(), sizeof(double));

     //NOTE: HERE is where the initial direction matrix is built. Change these values to change axis assignment
     for(int i=0; i < shape->getNumDims(); i++) {
       xi[i] = (double*)calloc(shape->getNumDims(), sizeof(double));
       xi[i][i] = 1.0;
     }
     //TEST axes
     xi[0][0] = 0.5;
     xi[0][1] = 0.5;

     //here, we will be using powell classes, and, instead of passing in calcDist, will be passing in a shape.
     shapeFound = powell(shape->getDimensions(), xi, shape->getNumDims(), 0.5, &iter, &fret, shape);
     if(shapeFound){
       p = shape->getDimensions();//TEST
       printf("Shape found!\n");
       printf("Final dimensions are -- ");
       for(int w = 1; w <=shape->getNumDims(); w++){
         printf("%d: %f ", w, p[w]);//outputs final values
       }
       printf("Certainty (lower is better): %f\n", fret);
     }else{
       printf("Shape not found -- certainty was %f\n", fret);
     }
   }
   //commenting out annoying outputs -- Z

   //printf("%f %f %f %f %f %f\n", p[0], p[1], p[2], p[3], p[4], fret);
   //sprintf(outfilename, "%s_d_%3.2f_l_%3.2f_h_%3.2f.pgm", infilename, sigma, tlow, thigh);
   //Raster::WriteFloat(distMap, FLOAT_NORM_0_255, outfilename);//distance transfer

   return(shapeFound);
}

void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb,
            double *fc, double (*func)(double, rutz::shared_ptr<ShapeModel>& ),
            rutz::shared_ptr<ShapeModel>& shape)
{
        double ulim,u,r,q,fu,dum;

        *fa=(*func)(*ax, shape);
        *fb=(*func)(*bx, shape);
        if (*fb > *fa) {
                SHFT(dum,*ax,*bx,dum)
                SHFT(dum,*fb,*fa,dum)
        }
        *cx=(*bx)+GOLD*(*bx-*ax);
        *fc=(*func)(*cx, shape);
        while (*fb > *fc) {
                r=(*bx-*ax)*(*fb-*fc);
                q=(*bx-*cx)*(*fb-*fa);
                u=(*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/
                        (2.0*SIGN(std::max(fabs(q-r),TINY),q-r));
                ulim=(*bx)+GLIMIT*(*cx-*bx);
                if ((*bx-u)*(u-*cx) > 0.0) {
                        fu=(*func)(u, shape);
                        if (fu < *fc) {
                                *ax=(*bx);
                                *bx=u;
                                *fa=(*fb);
                                *fb=fu;
                                return;
                        } else if (fu > *fb) {
                                *cx=u;
                                *fc=fu;
                                return;
                        }
                        u=(*cx)+GOLD*(*cx-*bx);
                        fu=(*func)(u, shape);
                } else if ((*cx-u)*(u-ulim) > 0.0) {
                        fu=(*func)(u, shape);
                        if (fu < *fc) {
                                SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx))
                                SHFT(*fb,*fc,fu,(*func)(u, shape))
                        }
                } else if ((u-ulim)*(ulim-*cx) >= 0.0) {
                        u=ulim;
                        fu=(*func)(u, shape);
                } else {
                        u=(*cx)+GOLD*(*cx-*bx);
                        fu=(*func)(u, shape);
                }
                SHFT(*ax,*bx,*cx,u)
                SHFT(*fa,*fb,*fc,fu)
        }
}

double brent(double ax, double bx, double cx, double (*f)(double, rutz::shared_ptr<ShapeModel>&), double tol,
        double *xmin, rutz::shared_ptr<ShapeModel>& shape)
{
        int iter;
        double a,b,d=0.0,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
        double e=0.0;

        a=(ax < cx ? ax : cx);
        b=(ax > cx ? ax : cx);
        x=w=v=bx;
        fw=fv=fx=(*f)(x, shape);
        for (iter=1;iter<=ITMAXB;iter++) {
                xm=0.5*(a+b);
                tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
                if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
                        *xmin=x;
                        return fx;
                }
                if (fabs(e) > tol1) {
                        r=(x-w)*(fx-fv);
                        q=(x-v)*(fx-fw);
                        p=(x-v)*q-(x-w)*r;
                        q=2.0*(q-r);
                        if (q > 0.0) p = -p;
                        q=fabs(q);
                        etemp=e;
                        e=d;
                        if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
                                d=CGOLD*(e=(x >= xm ? a-x : b-x));
                        else {
                                d=p/q;
                                u=x+d;
                                if (u-a < tol2 || b-u < tol2)
                                        d=SIGN(tol1,xm-x);
                        }
                } else {
                        d=CGOLD*(e=(x >= xm ? a-x : b-x));
                }
                u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
                fu=(*f)(u, shape);
                if (fu <= fx) {
                        if (u >= x) a=x; else b=x;
                        SHFT(v,w,x,u)
                        SHFT(fv,fw,fx,fu)
                } else {
                        if (u < x) a=u; else b=u;
                        if (fu <= fw || w == x) {
                                v=w;
                                w=u;
                                fv=fw;
                                fw=fu;
                        } else if (fu <= fv || v == x || v == w) {
                                v=u;
                                fv=fu;
                        }
                }
        }
        printf("Too many iterations in brent");
        *xmin=x;
        return fx;
}

//Changed to run dim on ShapeModel
double f1dim(double x, rutz::shared_ptr<ShapeModel>& shape)
{
        int j;
        double f, *xt;
        xt=nrVector(1,ncom);

        for (j=1;j<=ncom;j++) xt[j]=pcom[j]+x*xicom[j];
        f=shape->calcDist(xt, distMap);
        free_nrVector(xt,1,ncom);
        return f;
}

//Changed to accept ShapeModel optimizee class
void linmin(double p[], double xi[], int n, double *fret,
            rutz::shared_ptr<ShapeModel>& optimizee)
{
        int j;
        double xx,xmin,fx,fb,fa,bx,ax;

        ncom=n;
        pcom=nrVector(1,n);
        xicom=nrVector(1,n);
        for (j=1;j<=n;j++) {
                pcom[j]=p[j];
                xicom[j]=xi[j];
        }

        ax=0.0;
        xx=1.0;

        mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim,optimizee);
        *fret=brent(ax,xx,bx,f1dim,TOL,&xmin, optimizee);
        for (j=1;j<=n;j++) {
                xi[j] *= xmin;
                p[j] += xi[j];
        }
        free_nrVector(xicom,1,n);
        free_nrVector(pcom,1,n);
}

//Changed to accept ShapeModel optimizee class
bool powell(double p[], double **xi, int n, double ftol,
            int *iter, double *fret, rutz::shared_ptr<ShapeModel>& optimizee)
{
        int i,ibig,j;
        double del,fp,fptt,t, *pt, *ptt, *xit;

        pt=nrVector(1,n);
        ptt=nrVector(1,n);
        xit=nrVector(1,n);

        fptt = optimizee->getThreshold(); //hacky, to prevent warning

        *fret=optimizee->calcDist(p, distMap);
        for (j=1;j<=n;j++) pt[j]=p[j];
        for (*iter=1;;++(*iter)) {
                fp=(*fret);
                ibig=0;
                del=0.0;
                for (i=1;i<=n;i++) {
                        for (j = 1; j <= n; j++){
                          //NOTE: This stupid change using -1 for the xi vector is necessary to handle weirdness of the nrVectors
                          xit[j] = xi[j-1][i-1];
                        }
                        fptt=(*fret);
                        linmin(p,xit,n,fret,optimizee);
                        if (fabs(fptt-(*fret)) > del) {
                                del=fabs(fptt-(*fret));
                                ibig=i;
                        }
                }
                if (2.0*fabs(fp-(*fret)) <= ftol*(fabs(fp)+fabs(*fret))) {

                        //If cost is below threshold, then return that the shape was found
                        //perhaps this is not in the most ideal spot. --Z
                        if(fptt < optimizee->getThreshold()){/*if cost estimate is below threshold, shape is present in
                                           image (very rudimentary, need to take dimensions into account)*/
                          return(true);//test
                        }
                        //otherwise, return that it was not
                        else{return(false);}//test
                }
                if (*iter == ITMAX){
                  return(false);//test
                  printf("powell exceeding maximum iterations.");
                }
                for (j=1;j<=n;j++) {
                        ptt[j]=2.0*p[j]-pt[j];
                        xit[j]=p[j]-pt[j];
                        pt[j]=p[j];
                }
                fptt=optimizee->calcDist(ptt, distMap);
                if (fptt < fp) {
                  t=2.0*(fp-2.0*(*fret)+fptt)*squareOf(fp-(*fret)-del)-del*squareOf(fp-fptt);
                  if (t < 0.0) {
                    linmin(p,xit,n,fret,optimizee);
                    for (j=0;j<n;j++) {
                      xi[j][ibig]=xi[j][n];
                      xi[j][n]=xit[j];
                    }
                  }
                }

        }
}

/*******************************************************************************
* PROCEDURE: canny
* PURPOSE: To perform canny edge detection.
* NAME: Mike Heath
* DATE: 2/15/96
//Pradeep: returns the centroid of the "white" pixels
*******************************************************************************/
int canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, const char* fname)
{
   FILE *fpdir=NULL;          /* File to write the gradient image to.     */
   unsigned char *nms;        /* Points that are local maximal magnitude. */
   short int *smoothedim,     /* The image after gaussian smoothing.      */
             *delta_x,        /* The first devivative image, x-direction. */
             *delta_y,        /* The first derivative image, y-direction. */
             *magnitude;      /* The magnitude of the gadient image.      */
   float *dir_radians=NULL;   /* Gradient direction image.                */

   /****************************************************************************
   * Perform gaussian smoothing on the image using the input standard
   * deviation.
   ****************************************************************************/
   gaussian_smooth(image, rows, cols, sigma, &smoothedim);

   /****************************************************************************
   * Compute the first derivative in the x and y directions.
   ****************************************************************************/
   derrivative_x_y(smoothedim, rows, cols, &delta_x, &delta_y);

   /****************************************************************************
   * This option to write out the direction of the edge gradient was added
   * to make the information available for computing an edge quality figure
   * of merit.
   ****************************************************************************/
   if(fname != NULL){
      /*************************************************************************
      * Compute the direction up the gradient, in radians that are
      * specified counteclockwise from the positive x-axis.
      *************************************************************************/
      radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

      /*************************************************************************
      * Write the gradient direction image out to a file.
      *************************************************************************/
      if((fpdir = fopen(fname, "wb")) == NULL){
         fprintf(stderr, "Error opening the file %s for writing.\n", fname);
         exit(1);
      }
      fwrite(dir_radians, sizeof(float), rows*cols, fpdir);
      fclose(fpdir);
      free(dir_radians);
   }

   /****************************************************************************
   * Compute the magnitude of the gradient.
   ****************************************************************************/
   magnitude_x_y(delta_x, delta_y, rows, cols, &magnitude);

   /****************************************************************************
   * Perform non-maximal suppression.
   ****************************************************************************/
   if((nms = (unsigned char *) calloc(rows*cols,sizeof(unsigned char)))==NULL){
      fprintf(stderr, "Error allocating the nms image.\n");
      exit(1);
   }

   non_max_supp(magnitude, delta_x, delta_y, rows, cols, nms);

   /****************************************************************************
   * Use hysteresis to mark the edge pixels.
   ****************************************************************************/
   if((*edge=(unsigned char *)calloc(rows*cols,sizeof(unsigned char))) ==NULL){
      fprintf(stderr, "Error allocating the edge image.\n");
      exit(1);
   }
   //printf("\n%d %d %d %d %d %d\n", magnitude, nms, rows, cols, tlow, thigh);
     int centroid = apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);
   /****************************************************************************
   * Free all of the memory that we allocated except for the edge image that
   * is still being used to store out result.
   ****************************************************************************/
   free(smoothedim);
   free(delta_x);
   free(delta_y);
   free(magnitude);
   free(nms);
   return centroid;
}

/*******************************************************************************
* Procedure: radian_direction
* Purpose: To compute a direction of the gradient image from component dx and
* dy images. Because not all derriviatives are computed in the same way, this
* code allows for dx or dy to have been calculated in different ways.
*
* FOR X:  xdirtag = -1  for  [-1 0  1]
*         xdirtag =  1  for  [ 1 0 -1]
*
* FOR Y:  ydirtag = -1  for  [-1 0  1]'
*         ydirtag =  1  for  [ 1 0 -1]'
*
* The resulting angle is in radians measured counterclockwise from the
* xdirection. The angle points "up the gradient".
*******************************************************************************/
void radian_direction(short int *delta_x, short int *delta_y, int rows,
    int cols, float **dir_radians, int xdirtag, int ydirtag)
{
   int r, c, pos;
   float *dirim=NULL;
   double dx, dy;

   /****************************************************************************
   * Allocate an image to store the direction of the gradient.
   ****************************************************************************/
   if((dirim = (float *) calloc(rows*cols, sizeof(float))) == NULL){
      fprintf(stderr, "Error allocating the gradient direction image.\n");
      exit(1);
   }
   *dir_radians = dirim;

   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
         dx = (double)delta_x[pos];
         dy = (double)delta_y[pos];

         if(xdirtag == 1) dx = -dx;
         if(ydirtag == -1) dy = -dy;

         dirim[pos] = (float)angle_radians(dx, dy);
      }
   }
}

/*******************************************************************************
* FUNCTION: angle_radians
* PURPOSE: This procedure computes the angle of a vector with components x and
* y. It returns this angle in radians with the answer being in the range
* 0 <= angle <2*PI.
*******************************************************************************/
double angle_radians(double x, double y)
{
   double xu, yu, ang;

   xu = fabs(x);
   yu = fabs(y);

   if((xu == 0) && (yu == 0)) return(0);

   ang = atan(yu/xu);

   if(x >= 0){
      if(y >= 0) return(ang);
      else return(2*M_PI - ang);
   }
   else{
      if(y >= 0) return(M_PI - ang);
      else return(M_PI + ang);
   }
}

/*******************************************************************************
* PROCEDURE: magnitude_x_y
* PURPOSE: Compute the magnitude of the gradient. This is the square root of
* the sum of the squared derivative values.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
        short int **magnitude)
{
   int r, c, pos, sq1, sq2;

   /****************************************************************************
   * Allocate an image to store the magnitude of the gradient.
   ****************************************************************************/
   if((*magnitude = (short *) calloc(rows*cols, sizeof(short))) == NULL){
      fprintf(stderr, "Error allocating the magnitude image.\n");
      exit(1);
   }

   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
         sq1 = (int)delta_x[pos] * (int)delta_x[pos];
         sq2 = (int)delta_y[pos] * (int)delta_y[pos];
         (*magnitude)[pos] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
      }
   }

}

/*******************************************************************************
* PROCEDURE: derrivative_x_y
* PURPOSE: Compute the first derivative of the image in both the x any y
* directions. The differential filters that are used are:
*
*                                          -1
*         dx =  -1 0 +1     and       dy =  0
*                                          +1
*
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void derrivative_x_y(short int *smoothedim, int rows, int cols,
        short int **delta_x, short int **delta_y)
{
   int r, c, pos;

   /****************************************************************************
   * Allocate images to store the derivatives.
   ****************************************************************************/
   if(((*delta_x) = (short *) calloc(rows*cols, sizeof(short))) == NULL){
      fprintf(stderr, "Error allocating the delta_x image.\n");
      exit(1);
   }
   if(((*delta_y) = (short *) calloc(rows*cols, sizeof(short))) == NULL){
      fprintf(stderr, "Error allocating the delta_x image.\n");
      exit(1);
   }

   /****************************************************************************
   * Compute the x-derivative. Adjust the derivative at the borders to avoid
   * losing pixels.
   ****************************************************************************/
   for(r=0;r<rows;r++){
      pos = r * cols;
      (*delta_x)[pos] = smoothedim[pos+1] - smoothedim[pos];
      pos++;
      for(c=1;c<(cols-1);c++,pos++){
         (*delta_x)[pos] = smoothedim[pos+1] - smoothedim[pos-1];
      }
      (*delta_x)[pos] = smoothedim[pos] - smoothedim[pos-1];
   }

   /****************************************************************************
   * Compute the y-derivative. Adjust the derivative at the borders to avoid
   * losing pixels.
   ****************************************************************************/
   for(c=0;c<cols;c++){
      pos = c;
      (*delta_y)[pos] = smoothedim[pos+cols] - smoothedim[pos];
      pos += cols;
      for(r=1;r<(rows-1);r++,pos+=cols){
         (*delta_y)[pos] = smoothedim[pos+cols] - smoothedim[pos-cols];
      }
      (*delta_y)[pos] = smoothedim[pos] - smoothedim[pos-cols];
   }
}

/*******************************************************************************
* PROCEDURE: gaussian_smooth
* PURPOSE: Blur an image with a gaussian filter.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
        short int **smoothedim)
{
   int r, c, rr, cc,     /* Counter variables. */
      windowsize,        /* Dimension of the gaussian kernel. */
      center;            /* Half of the windowsize. */
   float *tempim,        /* Buffer for separable filter gaussian smoothing. */
         *kernel,        /* A one dimensional gaussian kernel. */
         dot,            /* Dot product summing variable. */
         sum;            /* Sum of the kernel weights variable. */

   /****************************************************************************
   * Create a 1-dimensional gaussian smoothing kernel.
   ****************************************************************************/
   make_gaussian_kernel(sigma, &kernel, &windowsize);
   center = windowsize / 2;

   /****************************************************************************
   * Allocate a temporary buffer image and the smoothed image.
   ****************************************************************************/
   if((tempim = (float *) calloc(rows*cols, sizeof(float))) == NULL){
      fprintf(stderr, "Error allocating the buffer image.\n");
      exit(1);
   }
   if(((*smoothedim) = (short int *) calloc(rows*cols,
         sizeof(short int))) == NULL){
      fprintf(stderr, "Error allocating the smoothed image.\n");
      exit(1);
   }

   /****************************************************************************
   * Blur in the x - direction.
   ****************************************************************************/
   for(r=0;r<rows;r++){
      for(c=0;c<cols;c++){
         dot = 0.0;
         sum = 0.0;
         for(cc=(-center);cc<=center;cc++){
            if(((c+cc) >= 0) && ((c+cc) < cols)){
               dot += (float)image[r*cols+(c+cc)] * kernel[center+cc];
               sum += kernel[center+cc];
            }
         }
         tempim[r*cols+c] = dot/sum;
      }
   }

   /****************************************************************************
   * Blur in the y - direction.
   ****************************************************************************/
   for(c=0;c<cols;c++){
      for(r=0;r<rows;r++){
         sum = 0.0;
         dot = 0.0;
         for(rr=(-center);rr<=center;rr++){
            if(((r+rr) >= 0) && ((r+rr) < rows)){
               dot += tempim[(r+rr)*cols+c] * kernel[center+rr];
               sum += kernel[center+rr];
            }
         }
         (*smoothedim)[r*cols+c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5);
      }
   }

   free(tempim);
   free(kernel);
}

/*******************************************************************************
* PROCEDURE: make_gaussian_kernel
* PURPOSE: Create a one dimensional gaussian kernel.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
   int i, center;
   float x, fx, sum=0.0;

   *windowsize = int(1 + 2 * ceil(2.5 * sigma));
   center = (*windowsize) / 2;

   if((*kernel = (float *) calloc((*windowsize), sizeof(float))) == NULL){
      fprintf(stderr, "Error callocing the gaussian kernel array.\n");
      exit(1);
   }

   for(i=0;i<(*windowsize);i++){
      x = (float)(i - center);
      fx = pow(2.71828, -0.5*x*x/(sigma*sigma)) / (sigma * sqrt(6.2831853));
      (*kernel)[i] = fx;
      sum += fx;
   }

   for(i=0;i<(*windowsize);i++) (*kernel)[i] /= sum;

}
