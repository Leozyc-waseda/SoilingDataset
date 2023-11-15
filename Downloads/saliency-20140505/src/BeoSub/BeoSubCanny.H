/*!@file BeoSub/BeoSubCanny.H A class for the vision(ported from test-canny) */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubCanny.H $
// $Id: BeoSubCanny.H 8491 2007-06-13 23:51:45Z beobot $
//

#ifndef BEOSUBCANNY_H_DEFINED
#define BEOSUBCANNY_H_DEFINED


/**************************TODO************************* -NOTE:
    switching to std vectors BREAKS powell, even if changed
    throughout. Note sure why

    -MAKE a tester executable to spool camera values (similar to
     those displayed at the start of test-multigrab) continually,
     checking white balance, exposure and gain in order to manually
     calibrate the cameras at wet test time. Auto set is not working
     well. NOTE that every color setting is currently rcognizing
     white! this is bad! something wrong with the V range? <-seems
     fixable using a mix of changing V and hardware stuff

    -SUGGESTION: instead of returning final recognized position data
     using a struct, inseat store data in shape class and return using
     reference. (allows shapes with different dimensions to return
     useful data, unlike current standard 5 dims)
*******************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "BeoSub/hysteresis.H"
#include "GUI/XWindow.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Transforms.H"
#include "Raster/Raster.H"
#include "VFAT/segmentImageTrackMC.H"
#include "rutz/shared_ptr.h"

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

#include "Component/ModelManager.H"
#include "Devices/IEEE1394grabber.H"
#include "Devices/V4Lgrabber.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>

//nrutil
#define FREE_ARG char*

//SHAPE STUFF
#include "BeoSub/CannyModel.H"

//! Definition and access functions for the BeoSubCanny
/*! BeoSubCanny is the vision class for submarine. It implements
    the Canny algorithm and can track shape and color*/
class BeoSubCanny : public ModelComponent
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  BeoSubCanny(OptionManager& mgr,
              const std::string& descrName = "BeoSubCanny",
              const std::string& tagName = "BeoSubCanny");

  //! Destructor
  ~BeoSubCanny();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{
  int canny(unsigned char *image, int rows, int cols, float sigma,
            float tlow, float thigh, unsigned char **edge, char *fname);
  void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
                       short int **smoothedim);
  void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
  void derrivative_x_y(short int *smoothedim, int rows, int cols,
                       short int **delta_x, short int **delta_y);
  void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
                     short int **magnitude);
  void radian_direction(short int *delta_x, short int *delta_y, int rows,
                        int cols, float **dir_radians, int xdirtag, int ydirtag);
  double angle_radians(double x, double y);
  void grabImage(Image<PixRGB<byte> >* image);

  void setupCanny(const char* colorArg, Image< PixRGB<byte> > image, bool debug);

  bool runCanny(rutz::shared_ptr<ShapeModel>& shapeArg); //FIX: change later so this returns struct with found info (position, etc)

  //From Numerical Recipes
  bool powell(double p[], double **xi, int n, double ftol,
              int *iter, double *fret, rutz::shared_ptr<ShapeModel>& optimizee);
  double brent(double ax, double bx, double cx, double tol,
               double *xmin, rutz::shared_ptr<ShapeModel>& shape);
  void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb,
              double *fc, rutz::shared_ptr<ShapeModel>& shape);
  double f1dim(double x, rutz::shared_ptr<ShapeModel>& shape);
  void linmin(double p[], double xi[], int n, double *fret,
              rutz::shared_ptr<ShapeModel>& optimizee);
  //nrutil stuff
  double *nrVector(long nl, long nh);
  void free_nrVector(double *v, long nl, long nh);

  //@}

private:

  Image<float> distMap;
  rutz::shared_ptr<XWindow> xwin, win;
  bool debugmode;

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

  segmentImageTrackMC<float,unsigned int,4> *segmenter;

  //! Mean color to track
  std::vector<float> color;

  int ncom;
  double *pcom,*xicom;
  bool hasSetup;

};


#endif
