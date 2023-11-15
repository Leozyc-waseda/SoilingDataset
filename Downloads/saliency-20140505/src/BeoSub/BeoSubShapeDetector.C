/*!@file BeoSub/BeoSubShapeDetector.C vision for the sub */

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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubShapeDetector.C $
// $Id: BeoSubShapeDetector.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "BeoSub/BeoSubShapeDetector.H"

#include "Image/MathOps.H"
#include "Image/MorphOps.H"     // for erodeImg(), closeImg()
#include "Image/Transforms.H"
#include "Util/MathFunctions.H" // for squareOf() inline function

#include <algorithm> // for std::max()

// ######################################################################
BeoSubCanny::BeoSubCanny(OptionManager& mgr,
                         const std::string& descrName,
                         const std::string& tagName):
  ModelComponent(mgr, descrName, tagName)
{
  //set math details for blur and threshold and such. For now, harcoded
  sigma = .5;
  tlow = .5;
  thigh = .5;

  color.resize(4, 0.0F);

  hasSetup = false;
  debugmode = true;
}

// ######################################################################
//FIX to accept color as an array rather than a char arg
void BeoSubCanny::setupCanny(const char* colorArg, Image< PixRGB<byte> > image, bool debug)
{

  bool useColor = true;

  debugmode = debug;//turns output on or off

  if(!hasSetup){
    segmenter = new segmentImageTrackMC<float,unsigned int,4> (image.getWidth()*image.getHeight());
    int wif = image.getWidth()/4;
    int hif = image.getHeight()/4;
    segmenter->SITsetFrame(&wif,&hif);
    // Set display colors for output of tracking. Strictly aesthetic //TEST
    //And definitely not needed, although for some reason it's required --Z
    segmenter->SITsetCircleColor(0,255,0);
    segmenter->SITsetBoxColor(255,255,0,0,255,255);

    segmenter->SITsetUseSmoothing(true,10);
    //method provided by Nathan to turn off adaptation (useful for dynamic environments)
    segmenter->SITtoggleColorAdaptation(false);
    //enable single-frame tracking
    segmenter->SITtoggleCandidateBandPass(false);
  }


  //COLORTRACKING DECS: note that much of these may not need to be in this file. Instead, the module could accept an array describing the color.

  //Colors MUST use H2SV2 pixel values! use test-sampleH2SV2 to sample needed values! --Z

  //NOTE: These color presets are extremely dependent on the camera hardware settings. It seems to be best to manually set the white blance and other settings.
 //0 = H1 (0-1), 1=H2 (0-1), 2=S (0-1), 3=V (0-1)
  if(!strcmp(colorArg, "Red")){
    //RED
    color[0] = 0.76700F; color[1] = 0.85300F; color[2] = 0.80000F; color[3] = 0.97200F;
  }

  else if(!strcmp(colorArg, "Green")){
    //Green
    color[0] = 0.53000F; color[1] = 0.36000F; color[2] = 0.44600F; color[3] = 0.92000F;
  }

  else if(!strcmp(colorArg, "Orange")){
    //"ORANGE"
    color[0] = 0.80000F; color[1] = 0.80000F; color[2] = 0.55000F; color[3] = 1.00000F;
  }
  else if(!strcmp(colorArg, "Blue")){
    //BLUE
    color[0] = 0.27500F; color[1] = 0.52500F; color[2] = 0.44500F; color[3] = 0.97200F;
  }
  else if(!strcmp(colorArg, "Yellow")){
    //YELLOW
    color[0] = 0.89000F; color[1] = 0.65000F; color[2] = 0.90000F; color[3] = 0.97000F;
  }
  else if(!strcmp(colorArg, "White")){
    color[0] = 0.70000F; color[1] = 0.70000F; color[2] = 0.04000F; color[3] = 1.00000F;
  }
  else if(!strcmp(colorArg, "Black")){
    //color[0] = 0.60001F; color[1] = 0.66000F; color[2] = 0.24000F; color[3] = 0.70000F;
color[0] = 0.79F; color[1] = 0.18F; color[2] = 0.22F; color[3] = 0.175F;
  }
  else{
    useColor = false;
  }

  //! +/- tollerance value on mean for track
  std::vector<float> std(4,0.0F);
  //NOTE that the saturation tolerance is important (if it goes any higher than this, it will nearly always recognize white!)
  std[0] = 0.30000; std[1] = 0.30000; std[2] = 0.34500; std[3] = 0.15000;//3 was .45, 2 was .445

  if(!strcmp(colorArg, "White") || !strcmp(colorArg, "Black")){
    std[0] = 1.0F; std[1] = 1.0F; std[2] = 0.1F; std[3] = 0.1F;
   }

   //! normalizer over color values (highest value possible)
   std::vector<float> norm(4,0.0F);
   norm[0] = 1.0F; norm[1] = 1.0F; norm[2] = 1.0F; norm[3] = 1.0F;

   //! how many standard deviations out to adapt, higher means less bias
   //The lower these are, the more strict recognition will be in subsequent frames
   //TESTED AND PROVED do NOT change unless you're SURE
   std::vector<float> adapt(4,0.0F);
   //adapt[0] = 3.5F; adapt[1] = 3.5F; adapt[2] = 3.5F; adapt[3] = 3.5F;
   adapt[0] = 5.0F; adapt[1] = 5.0F; adapt[2] = 5.0F; adapt[3] = 5.0F;

   if(!strcmp(colorArg, "White") || !strcmp(colorArg, "Black")){
     adapt[0] = 25.0F; adapt[1] = 25.0F; adapt[2] = 0.1F; adapt[3] = 0.1F;

   }

   //! highest value for color adaptation possible (hard boundry)
   std::vector<float> upperBound(4,0.0F);
   upperBound[0] = color[0] + 0.45F; upperBound[1] = color[1] + 0.45F; upperBound[2] = color[2] + 0.55F; upperBound[3] = color[3] + 0.55F;

   //! lowest value for color adaptation possible (hard boundry)
   std::vector<float> lowerBound(4,0.0F);
   lowerBound[0] = color[0] - 0.45F; lowerBound[1] = color[1] - 0.45F; lowerBound[2] = color[2] - 0.55F; lowerBound[3] = color[3] - 0.55F;
   //END DECS

   colorImg = image;
   imgW = colorImg.getWidth();
   imgH = colorImg.getHeight();

   //color tracking stuff
   segmenter->SITsetTrackColor(&color,&std,&norm,&adapt,&upperBound,&lowerBound);

   ww = imgW/4;
   hh = imgH/4;

   Image<PixRGB<byte> > Aux;
   Aux.resize(100,450,true);
   Image<byte> temp;
   Image< PixRGB<byte> > display = colorImg;

   if(debugmode){
     if(!win.get()){
       win.reset( new XWindow(colorImg.getDims(), -1, -1, "test-canny window") );
       win->setPosition(0, 0);
     }
     win->drawImage(colorImg);
   }

   //Perform color exclusion, if desired
   if(useColor){

     //Get candidate image representing blobs of desired color
     H2SVimage = colorImg;//changed since transfer to H2SV
     display = colorImg;
     segmenter->SITtrackImageAny(H2SVimage,&display,&Aux,true);
     temp = segmenter->SITreturnCandidateImage();//get image we'll be using for edge detection
     //NOTE: This line works off of a "weeded-out" version of the blob tracker. It may be best to switch to this once the colors are set VERY well. Unfortunately, it is picky enough about colors to make the color settings difficult to determine. FIX?
     //temp = (Image<byte>)(segmenter->SITreturnBlobMap());

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
     //temp = closeImg(temp, se2);
     /*//the following is to exclude non-surrounded colors. was used for the bin
     // flood from corners:
     Image<byte> temp2 = binaryReverse(temp, byte(255));
     floodClean(temp, temp2, Point2D<int>(0, 0), byte(255), byte(0));

     temp = temp2;
     floodClean(temp2, temp, Point2D<int>(temp.getWidth()-1, temp.getHeight()-1), byte(255), byte(0));

     temp = binaryReverse(temp, byte(255));
     */
     grayImg = (Image<byte>)(quickInterpolate(temp,4));//note that this is very low rez
   }
   else{
     grayImg = luminance(colorImg);
   }

   hasSetup = true;
}

// ######################################################################
bool BeoSubCanny::runCanny(rutz::shared_ptr<ShapeModel>& shapeArg){


  //Perform canny edge detection

 canny(grayImg.getArrayPtr(), grayImg.getHeight(), grayImg.getWidth(), sigma, tlow, thigh, &edge, NULL);

  Image<unsigned char> cannyCharImg(edge, grayImg.getWidth(), grayImg.getHeight());

  Image<float> cannyImg = cannyCharImg;

  Point2D<int> cent = centroid(cannyCharImg);
  distMap = chamfer34(cannyImg, 5000.0f);

 distMap = grayImg;

  //Raster::WriteGray(cannyImg, outfilename, RASFMT_PNM);//edge image
  if(debugmode){
    if(!xwin.get()){
      xwin.reset(new XWindow(colorImg.getDims()));
      xwin->setPosition(colorImg.getWidth()+10, 0);
    }
    xwin->drawImage((Image< PixRGB<byte> >)grayImg);
  }



  //perform shape detection
  //get initial dimensions for the shape

  bool shapeFound = false;
  double** xi = (double**)calloc(6, sizeof(double));


   //NOTE: HERE is where the initial direction matrix is built. Change these values to change axis assignment
   for(int i=0; i<6; i++) {
     xi[i] = (double*)calloc(6, sizeof(double));
     xi[i][i] = 1.0;
   }
   //Reset initial axes
   xi[0][0] = 0.5;
   xi[0][1] = 0.5;

  int iter = 0;
  double fret = 0.0;

  //here, we will be using powell classes, and, instead of passing in calcDist, will be passing in a shape.
   shapeFound = powell(shapeArg->getDimensions(), xi, shapeArg->getNumDims(), 0.5, &iter, &fret, shapeArg);

   if(debugmode){
     double* p = shapeArg->getDimensions();//TEST
     if(shapeFound){
       printf("\n\nShape found!\n");
       printf("Final dimensions are -- ");
       for(int w = 1; w <=shapeArg->getNumDims(); w++){
         printf("%d: %f ", w, p[w]);//outputs final values
       }
       printf("Certainty (lower is better): %f\n\n", fret);
     }else{
       printf("\n\nShape not found. Certainty was %f\n\n", fret);
     }
     }

   return(shapeFound);
}



// ######################################################################
BeoSubCanny::~BeoSubCanny()
{
}

///////////////////////FROM NRUTIL/////////////////////////
// ######################################################################
double *BeoSubCanny::nrVector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
        double *v;

        v=(double *)malloc((size_t) ((nh-nl+2)*sizeof(double)));
        if (!v) printf("allocation failure in nrVector()");
        return v-nl+1;
}

// ######################################################################
void BeoSubCanny::free_nrVector(double *v, long nl, long nh)
/* free a double vector allocated with nrVector() */
{
        free((FREE_ARG) (v+nl-1));
}


// ######################################################################
void BeoSubCanny::mnbrak(double *ax, double *bx, double *cx, double *fa,
                         double *fb, double *fc,
                         rutz::shared_ptr<ShapeModel>& shape)
{
  double ulim,u,r,q,fu,dum;

  *fa=f1dim(*ax, shape);
  *fb=f1dim(*bx, shape);
  if (*fb > *fa) {
    SHFT(dum,*ax,*bx,dum)
      SHFT(dum,*fb,*fa,dum)
      }
  *cx=(*bx)+GOLD*(*bx-*ax);
  *fc=f1dim(*cx, shape);
  while (*fb > *fc) {
    r=(*bx-*ax)*(*fb-*fc);
    q=(*bx-*cx)*(*fb-*fa);
    u=(*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/
      (2.0*SIGN(std::max(fabs(q-r),TINY),q-r));
    ulim=(*bx)+GLIMIT*(*cx-*bx);
    if ((*bx-u)*(u-*cx) > 0.0) {
      fu=f1dim(u, shape);
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
      fu=f1dim(u, shape);
    } else if ((*cx-u)*(u-ulim) > 0.0) {
      fu=f1dim(u, shape);
      if (fu < *fc) {
        SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx))
          SHFT(*fb,*fc,fu,f1dim(u, shape))
          }
    } else if ((u-ulim)*(ulim-*cx) >= 0.0) {
      u=ulim;
      fu=f1dim(u, shape);
    } else {
      u=(*cx)+GOLD*(*cx-*bx);
      fu=f1dim(u, shape);
    }
    SHFT(*ax,*bx,*cx,u)
      SHFT(*fa,*fb,*fc,fu)
      }
}

double BeoSubCanny::brent(double ax, double bx, double cx,
                          double tol, double *xmin,
                          rutz::shared_ptr<ShapeModel>& shape)
{
        int iter;
        double a,b,d=0.0,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
        double e=0.0;

        a=(ax < cx ? ax : cx);
        b=(ax > cx ? ax : cx);
        x=w=v=bx;
        fw=fv=fx=f1dim(x, shape);
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
                fu=f1dim(u, shape);
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

double BeoSubCanny::f1dim(double x, rutz::shared_ptr<ShapeModel>& shape)
{
        int j;
        double f, *xt;
        xt=nrVector(1,ncom);

        for (j=1;j<=ncom;j++) xt[j]=pcom[j]+x*xicom[j];
        f=shape->calcDist(xt, distMap);
        free_nrVector(xt,1,ncom);
        return f;
}

void BeoSubCanny::linmin(double p[], double xi[], int n, double *fret,
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

          mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,optimizee);
          *fret=brent(ax,xx,bx,TOL,&xmin, optimizee);

        for (j=1;j<=n;j++) {
                xi[j] *= xmin;
                p[j] += xi[j];
        }
        free_nrVector(xicom,1,n);
        free_nrVector(pcom,1,n);
}

bool BeoSubCanny::powell(double p[], double **xi, int n, double ftol,
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

                  //optimizee->setDimensions(p);//TEST

                  //If cost is below threshold, then return that the shape was found
                  //perhaps this is not in the most ideal spot. --Z
                  if(fptt < optimizee->getThreshold()){//if cost estimate is below threshold, shape is present in image (very rudimentary, need to take dimensions into account)
                          return(true);//test
                        }
                        //otherwise, return that it was not
                        else{return(false);}//test
                }
                if (*iter == ITMAX){
                  //optimizee->setDimensions(p);//TEST
                  printf("powell exceeding maximum iterations.");
                  return(false);//test
                }
                for (j=1;j<=n;j++) {
                        ptt[j]=2.0*p[j]-pt[j];
                        xit[j]=p[j]-pt[j];
                        pt[j]=p[j];
                }
                //NOTE: in order to fix problem of things becoming too small or going outside of the image, either put a check right here or check inside calc dist. Then, if values cause trouble, REVERT to values of things in above for loop that were set BEFORE the previous change. This may not help, however, depending on how calcDist is called in its first instance in this function
                //NOTE: it may be better instead to have the calcDist function return values by reference. In that case, a shape could correct its own dimensions if they went out of whack. Unfortunately, this would not work if poweel would still go down the wrong path. FIX!
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
int BeoSubCanny::canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname)
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
void BeoSubCanny::radian_direction(short int *delta_x, short int *delta_y, int rows,
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
double BeoSubCanny::angle_radians(double x, double y)
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
void BeoSubCanny::magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
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
void BeoSubCanny::derrivative_x_y(short int *smoothedim, int rows, int cols,
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
void BeoSubCanny::gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
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
void BeoSubCanny::make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
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
