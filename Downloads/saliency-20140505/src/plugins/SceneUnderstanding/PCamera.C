/*!@file SceneUnderstanding/PCamera.C  */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/PCamera.C $
// $Id: PCamera.C 13551 2010-06-10 21:56:32Z itti $
//

#ifndef PCamera_C_DEFINED
#define PCamera_C_DEFINED

#include "plugins/SceneUnderstanding/PCamera.H"

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Convolutions.H"
#include "Image/fancynorm.H"
#include "Image/Point3D.H"
#include "Image/MatrixOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/MathOps.H"
#include "Simulation/SimEventQueue.H"
#include "Neuro/EnvVisualCortex.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_PCamera = {
  MOC_SORTPRI_3,   "PCamera-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_PCameraShowDebug =
  { MODOPT_ARG(bool), "PCameraShowDebug", &MOC_PCamera, OPTEXP_CORE,
    "Show debug img",
    "pcamera-debug", '\0', "<true|false>", "false" };


#define KEY_UP 98
#define KEY_DOWN 104
#define KEY_LEFT 100
#define KEY_RIGHT 102

namespace
{

  // helper function:
  // finds a cosine of angle between vectors
  // from pt0->pt1 and from pt0->pt2
  double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 )
  {
    double dx1 = pt1->x - pt0->x;
    double dy1 = pt1->y - pt0->y;
    double dx2 = pt2->x - pt0->x;
    double dy2 = pt2->y - pt0->y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
  }
}


// ######################################################################
PCamera::PCamera(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventInputFrame),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  itsShowDebug(&OPT_PCameraShowDebug, this),
  itsCameraCtrl(new Visca(mgr)),
  itsStorage(cvCreateMemStorage(0))
{
  ASSERT(itsStorage != 0);

  addSubComponent(itsCameraCtrl);


  itsZoomOut = false;

        itsIntrinsicMatrix = cvCreateMat( 3, 3, CV_32FC1);
        itsDistortionCoeffs = cvCreateMat( 4, 1, CV_32FC1);
  itsCameraRotation = cvCreateMat( 1, 3, CV_64FC1);
  itsCameraTranslation = cvCreateMat( 1, 3, CV_64FC1);

//  cvmSet(itsDistortionCoeffs, 0, 0, 0.1237883  );
//  cvmSet(itsDistortionCoeffs, 1, 0, -26.7828178 );
//  cvmSet(itsDistortionCoeffs, 2, 0, -0.0112169  );
//  cvmSet(itsDistortionCoeffs, 3, 0, 0.0065743  );
// // cvmSet(itsDistortionCoeffs, 0, 0, 0 );
// // cvmSet(itsDistortionCoeffs, 1, 0, 0);
// // cvmSet(itsDistortionCoeffs, 2, 0, 0);
// // cvmSet(itsDistortionCoeffs, 3, 0, 0);
//
//  cvmSet(itsIntrinsicMatrix, 0, 0, 672.84222); cvmSet(itsIntrinsicMatrix, 0, 1, 0); cvmSet(itsIntrinsicMatrix, 0, 2, 159.50000);
//  cvmSet(itsIntrinsicMatrix, 1, 0, 0); cvmSet(itsIntrinsicMatrix, 1, 1, 3186.79102 ); cvmSet(itsIntrinsicMatrix, 1, 2, 119.5);
//  cvmSet(itsIntrinsicMatrix, 2, 0, 0); cvmSet(itsIntrinsicMatrix, 2, 1, 0); cvmSet(itsIntrinsicMatrix, 2, 2, 1);
//
//  //Set the extrensic params
  cvmSet(itsCameraRotation, 0, 0, 2.373648 );
  cvmSet(itsCameraRotation, 0, 1, -0.017453 );
  cvmSet(itsCameraRotation, 0, 2, 0.157079);
//
  cvmSet(itsCameraTranslation, 0, 0, 24);
  cvmSet(itsCameraTranslation, 0, 1, -21);
  cvmSet(itsCameraTranslation, 0, 2,  990.2);

  cvmSet(itsDistortionCoeffs, 0, 0, 0.0  );
  cvmSet(itsDistortionCoeffs, 1, 0, 0.0 );
  cvmSet(itsDistortionCoeffs, 2, 0, 0.0  );
  cvmSet(itsDistortionCoeffs, 3, 0, 0.0  );

  //double cameraParam[3][4] = {
  //  {350.475735, 0, 158.250000, 0},
  //  {0.000000, -363.047091, 118.250000, 0.000000},
  //  {0.000000, 0.000000, 1.000000, 0.00000}};

  cvmSet(itsIntrinsicMatrix, 0, 0,350.475735); cvmSet(itsIntrinsicMatrix, 0, 1, 0); cvmSet(itsIntrinsicMatrix, 0, 2, 158.25000);
  cvmSet(itsIntrinsicMatrix, 1, 0, 0); cvmSet(itsIntrinsicMatrix, 1, 1, -363.047091 ); cvmSet(itsIntrinsicMatrix, 1, 2, 118.25);
  cvmSet(itsIntrinsicMatrix, 2, 0, 0); cvmSet(itsIntrinsicMatrix, 2, 1, 0); cvmSet(itsIntrinsicMatrix, 2, 2, 1);

  itsIntrinsicInit = false;
  itsSaveCorners = false;
  itsDrawGrid = false;
  itsChangeRot = false;

  itsVP = new ViewPort3D(320,240);

 // itsCameraCtrl->zoom(600);
 // itsCameraCtrl->movePanTilt(0,0);

}

// ######################################################################
PCamera::~PCamera()
{

}

void PCamera::onSimEventInputFrame(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventInputFrame>& e)
{
  // here is the inputs image:
  GenericFrame frame = e->frame();

  const Image<PixRGB<byte> > inimg = frame.asRgb();
  itsCurrentImg = inimg;

  //Calibrate
  findSquares(inimg,
      itsStorage,
      2000,
      10000,
      0.6);

  //const Image<PixRGB<byte> > out = drawSquares(inimg, cards);

  //SHOWIMG(out);
  //getTransMat();

  //itsVP->setCamera(Point3D<float>(0,-460,300), Point3D<float>(-58,0,0));
  double trans[3][4] = {
    {-0.996624, 0.070027, 0.042869, -16.907477},
    {-0.004359, 0.476245, -0.879302, 9.913470},
    {-0.081990, -0.876520, -0.474332, 276.648010}};
  itsVP->setCamera(trans);



  itsVP->initFrame();
  itsVP->drawBox(
      Point3D<float>(0,0,20),
      Point3D<float>(0,0,0),
      Point3D<float>(40,40,40),
      PixRGB<byte>(0,256,0));

  itsRenderImg = flipVertic(itsVP->getFrame());


}

void PCamera::findSquares(const Image<PixRGB<byte> >& in, CvMemStorage* storage,
    const int minarea, const int maxarea, const double mincos)
{
  const int N = 11;

  itsSquares.clear();
  IplImage* img = img2ipl(in);

  CvSize sz = cvSize( img->width & -2, img->height & -2 );
  IplImage* timg = cvCloneImage( img ); // make a copy of input image
  IplImage* gray = cvCreateImage( sz, 8, 1 );
  IplImage* pyr = cvCreateImage( cvSize(sz.width/2, sz.height/2), 8, 3 );
  // create empty sequence that will contain points -
  // 4 points per square (the square's vertices)

  // select the maximum ROI in the image
  // with the width and height divisible by 2
  cvSetImageROI( timg, cvRect( 0, 0, sz.width, sz.height ));

  // down-scale and upscale the image to filter out the noise
  cvPyrDown( timg, pyr, 7 );
  cvPyrUp( pyr, timg, 7 );
  IplImage* tgray = cvCreateImage( sz, 8, 1 );

  // find squares in every color plane of the image
  for (int c = 0; c < 3; ++c)
  {
    // extract the c-th color plane
    cvSetImageCOI( timg, c+1 );
    cvCopy( timg, tgray, 0 );

    // try several threshold levels
    for (int l = 0; l < N; ++l)
    {
      // hack: use Canny instead of zero threshold level.
      // Canny helps to catch squares with gradient shading
      if( l == 0 )
      {
        // apply Canny. Take the upper threshold from slider
        // and set the lower to 0 (which forces edges merging)
        const int thresh = 50;
        cvCanny( tgray, gray, 0, thresh, 5 );
        // dilate canny output to remove potential
        // holes between edge segments
        cvDilate( gray, gray, 0, 1 );
      }
      else
      {
        // apply threshold if l!=0:
        //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
        cvThreshold( tgray, gray, (l+1)*255/N, 255, CV_THRESH_BINARY );
      }

      // find contours and store them all as a list
      CvSeq* contours = 0;
      cvFindContours( gray, storage, &contours, sizeof(CvContour),
          CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );

      // test each contour
      while( contours )
      {
        // approximate contour with accuracy proportional
        // to the contour perimeter
        CvSeq* result =
          cvApproxPoly( contours, sizeof(CvContour), storage,
              CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );
        // square contours should have 4 vertices after approximation
        // relatively large area (to filter out noisy contours)
        // and be convex.
        // Note: absolute value of an area is used because
        // area may be positive or negative - in accordance with the
        // contour orientation
        const double area = fabs(cvContourArea(result,CV_WHOLE_SEQ));
        if (result->total == 4 &&
            area >= minarea && area <= maxarea &&
            cvCheckContourConvexity(result))
        {
          double s = 0;

          for (int i = 0; i < 4; ++i)
          {
            // find minimum angle between joint
            // edges (maximum of cosine)
            const double t =
              fabs(angle((CvPoint*)cvGetSeqElem( result, i % 4 ),
                    (CvPoint*)cvGetSeqElem( result, (i-2) % 4 ),
                    (CvPoint*)cvGetSeqElem( result, (i-1) % 4 )));
            s = s > t ? s : t;
          }


          // if cosines of all angles are small
          // (all angles are ~90 degree) then write quandrangle
          // vertices to resultant sequence
          if (s < mincos)
          {
            PCamera::Square sq;
            for (int i = 0; i < 4; ++i)
            {
              CvPoint *pt = (CvPoint*)cvGetSeqElem( result, i );
              sq.p[i] = Point2D<int>(pt->x, pt->y);
            }
            itsSquares.push_back(sq);
            //  LINFO("area=%f, mincos=%f", area, s);
          }
        }

        // take the next contour
        contours = contours->h_next;
      }
    }
  }

  // release all the temporary images
  cvReleaseImage( &gray );
  cvReleaseImage( &pyr );
  cvReleaseImage( &tgray );
  cvReleaseImage( &timg );
  cvReleaseImageHeader( &img );

}


void PCamera::getTransMat()
{

  if (itsSquares.size() > 0)
  {
    CvMat *image_points_ex = cvCreateMat( 4, 2, CV_64FC1);

    for (uint j = 0; j < 4; j++){
      cvSetReal2D( image_points_ex, j, 0, itsSquares[0].p[j].i);
      cvSetReal2D( image_points_ex, j, 1, itsSquares[0].p[j].j);
    }

    CvMat *object_points_ex = cvCreateMat( 4, 3, CV_64FC1);
    //for (uint j = 0; j < corners.size(); j++){
    cvSetReal2D( object_points_ex, 0, 0, -40 ); cvSetReal2D( object_points_ex, 0, 1, -40 ); cvSetReal2D( object_points_ex, 0, 2, 0.0 );
    cvSetReal2D( object_points_ex, 1, 0, 40 );  cvSetReal2D( object_points_ex, 1, 1, -40 ); cvSetReal2D( object_points_ex, 1, 2, 0.0 );
    cvSetReal2D( object_points_ex, 2, 0, -40 ); cvSetReal2D( object_points_ex, 2, 1, 40 ); cvSetReal2D( object_points_ex, 2, 2, 0.0 );
    cvSetReal2D( object_points_ex, 3, 0, 40 );  cvSetReal2D( object_points_ex, 3, 1, 40 ); cvSetReal2D( object_points_ex, 3, 2, 0.0 );
    //}

    cvFindExtrinsicCameraParams2( object_points_ex,
        image_points_ex,
        itsIntrinsicMatrix,
        itsDistortionCoeffs,
        itsCameraRotation,
        itsCameraTranslation);

    printf( "Rotation2: %f %f %f\n",
        cvGetReal2D(itsCameraRotation, 0, 0),
        cvGetReal2D(itsCameraRotation, 0, 1),
        cvGetReal2D(itsCameraRotation, 0, 2));
    printf( "Translation2: %f %f %f\n",
        cvGetReal2D(itsCameraTranslation, 0, 0),
        cvGetReal2D(itsCameraTranslation, 0, 1),
        cvGetReal2D(itsCameraTranslation, 0, 2));
  }

}





//// ######################################################################
//void PCamera::onSimEventLGNOutput(SimEventQueue& q,
//                                  rutz::shared_ptr<SimEventLGNOutput>& e)
//{
//  itsPCameraCellsInput = e->getCells();
//
//
//  inplaceNormalize(itsPCameraCellsInput[0], 0.0F, 255.0F);
//  Image<byte> in = itsPCameraCellsInput[0];
//
//  int rows = 7;
//  int cols = 7;
//  std::vector<CvPoint2D32f> corners(rows*cols);
//
//  int count = 0;
//  int result = cvFindChessboardCorners(img2ipl(in), cvSize(rows,cols),
//      &corners[0], &count,
//      CV_CALIB_CB_ADAPTIVE_THRESH |
//      CV_CALIB_CB_NORMALIZE_IMAGE |
//      CV_CALIB_CB_FILTER_QUADS);
//
//  // result = 0 if not all corners were found
//  // Find corners to an accuracy of 0.1 pixel
//        if(result != 0)
//                cvFindCornerSubPix(img2ipl(in),
//        &corners[0],
//        count,
//        cvSize(10,10), //win
//        cvSize(-1,-1), //zero_zone
//        cvTermCriteria(CV_TERMCRIT_ITER,1000,0.01) );
//
//  //LINFO("Found %i %lu complete %i", count, corners.size(), result);
//  //cvDrawChessboardCorners(img2ipl(in), cvSize(rows,cols), &corners[0], count, result);
//
//        if(result != 0)
//  {
//    if (itsSaveCorners)
//    {
//      for(uint i=0; i<corners.size(); i++)
//        itsCorners.push_back(corners[i]);
//      itsSaveCorners=false;
//    }
//  }
//
//  //Set the intrinsnic params based on the zoom
////  float currentZoom = itsCameraCtrl->getCurrentZoom();
////  cvmSet(itsIntrinsicMatrix, 0, 0, 3.6e-06*pow(currentZoom, 3) - 0.0012*pow(currentZoom,2) + 0.9265*currentZoom + 415.5);
////  cvmSet(itsIntrinsicMatrix, 1, 1, 3.6e-06*pow(currentZoom, 3) - 0.0013*pow(currentZoom,2) + 0.9339*currentZoom + 436);
//
//
//  itsDebugImg = in;
//
//  std::vector<Point3D<float> > points;
//
//  //points.push_back(Point3D<float>(0,0,0));
//  //points.push_back(Point3D<float>(63.5,0,0));
//  //points.push_back(Point3D<float>(63.5,44.45,0));
//  //points.push_back(Point3D<float>(0,44.45,0));
//
//  //points.push_back(Point3D<float>(0,0,-50.8));
//  //points.push_back(Point3D<float>(63.5,0,-50.8));
//  //points.push_back(Point3D<float>(63.5,44.45,-50.8));
//  //points.push_back(Point3D<float>(0,44.45,-50.8));
//
//  Point3D<float> pos(-15,-15,0);
//  points.push_back(Point3D<float>(pos.x,pos.y,0));
//  points.push_back(Point3D<float>(pos.x,pos.y,30));
//  points.push_back(Point3D<float>(pos.x+30,pos.y,30));
//  points.push_back(Point3D<float>(pos.x+30,pos.y,0));
//  points.push_back(Point3D<float>(pos.x,pos.y,0));
//
//  points.push_back(Point3D<float>(pos.x,pos.y+30,0));
//  points.push_back(Point3D<float>(pos.x,pos.y+30,30));
//  points.push_back(Point3D<float>(pos.x,pos.y,30));
//
//
//  points.push_back(Point3D<float>(pos.x+30,pos.y,30));
//  points.push_back(Point3D<float>(pos.x+30,pos.y+30,30));
//  points.push_back(Point3D<float>(pos.x,pos.y+30,30));
//  points.push_back(Point3D<float>(pos.x,pos.y,30));
//
//
//
//  //points.push_back(Point3D<float>(30,30,0));
//  //points.push_back(Point3D<float>(0,30,0));
//
//  //points.push_back(Point3D<float>(0,0,-4.5));
//  //points.push_back(Point3D<float>(30,0,-4.5));
//  //points.push_back(Point3D<float>(30,30,-4.5));
//  //points.push_back(Point3D<float>(0,30,-4.5));
//
//  if (itsDrawGrid)
//    projectGrid();
//  //  displayExtrinsic(corners);
// //projectPoints(points);
//
//  //double d0 = cvGetReal2D( itsDistortionCoeffs, 0, 0);
//  //double d1 = cvGetReal2D( itsDistortionCoeffs, 1, 0);
//  //double d2 = cvGetReal2D( itsDistortionCoeffs, 2, 0);
//  //double d3 = cvGetReal2D( itsDistortionCoeffs, 3, 0);
//  ////printf( "distortion_coeffs ( %7.7lf, %7.7lf, %7.7lf, %7.7lf)\n", d0, d1, d2, d3);
//
//  //double ir00 = cvGetReal2D( itsIntrinsicMatrix, 0, 0);
//  //double ir01 = cvGetReal2D( itsIntrinsicMatrix, 0, 1);
//  //double ir02 = cvGetReal2D( itsIntrinsicMatrix, 0, 2);
//  //double ir10 = cvGetReal2D( itsIntrinsicMatrix, 1, 0);
//  //double ir11 = cvGetReal2D( itsIntrinsicMatrix, 1, 1);
//  //double ir12 = cvGetReal2D( itsIntrinsicMatrix, 1, 2);
//  //double ir20 = cvGetReal2D( itsIntrinsicMatrix, 2, 0);
//  //double ir21 = cvGetReal2D( itsIntrinsicMatrix, 2, 1);
//  //double ir22 = cvGetReal2D( itsIntrinsicMatrix, 2, 2);
//  //printf( "intrinsics ( %7.5lf, %7.5lf, %7.5lf)\n", ir00, ir01, ir02);
//  //printf( "           ( %7.5lf, %7.5lf, %7.5lf)\n", ir10, ir11, ir12);
//  //printf( "           ( %7.5lf, %7.5lf, %7.5lf)\n", ir20, ir21, ir22);
//
//  //printf( "Rotation: %f %f %f\n",
//  //    cvGetReal2D(itsCameraRotation, 0, 0),
//  //    cvGetReal2D(itsCameraRotation, 0, 1),
//  //    cvGetReal2D(itsCameraRotation, 0, 2));
//  //printf( "Translation: %f %f %f\n",
//  //    cvGetReal2D(itsCameraTranslation, 0, 0),
//  //    cvGetReal2D(itsCameraTranslation, 0, 1),
//  //    cvGetReal2D(itsCameraTranslation, 0, 2));
//
//}

// ######################################################################
void PCamera::onSimEventPTZToLoc(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventUserInput>& e)
{

  Point2D<int> targetLoc = e->getMouseClick();

  //LINFO("PTZ to %ix%i",
  //    targetLoc.i,
  //    targetLoc.j);

  //if (targetLoc.isValid())
  //{
  //  itsZoomOut = false;
  //  q.post(rutz::make_shared(new SimEventSetVisualTracker(this, targetLoc)));
  //}

}

//void PCamera::onSimEventVisualTracker(SimEventQueue& q, rutz::shared_ptr<SimEventVisualTracker>& e)
//{
//
//  if (e->isTracking())
//  {
//    itsCurrentTargetLoc = e->getTargetLoc();
//
//    //move camera to target loc
//    Point2D<int> locErr = itsCurrentTargetLoc - Point2D<int>(320/2, 240/2);
//
//    int currentZoom = itsCameraCtrl->getCurrentZoom();
//
//    Point2D<float> pGain(0.15, 0.15);
//
//    if (currentZoom > 800)
//    {
//      pGain.i = 0.05; pGain.j = 0.06;
//    } else if (currentZoom > 1000)
//    {
//      pGain.i = 0.04; pGain.j = 0.04;
//    }
//
//
//    //P controller for now
//    Point2D<float> u = pGain*locErr;
//
//
//    LINFO("***Target is at: %ix%i zoom=%i (err %ix%i) move=%ix%i",
//        itsCurrentTargetLoc.i, itsCurrentTargetLoc.j, currentZoom,
//        locErr.i, locErr.j,
//        (int)u.i, (int)u.j);
//
//    if (fabs(locErr.distance(Point2D<int>(0,0))) > 16)
//      itsCameraCtrl->movePanTilt((int)u.i, -1*(int)u.j, true); //move relative
//    else
//    {
//      LINFO("****** zoom out");
//      if (itsZoomOut)
//        itsCameraCtrl->zoom(400); //Zoom out
//      else
//        itsCameraCtrl->zoom(10, true); //move relative
//    }
//  } else {
//    //LINFO("Not tracking");
//  }
//
//
//}


void PCamera::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event %s %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  onSimEventPTZToLoc(q, e);

  bool moveCamera = true;

  switch(e->getKey())
  {
    case KEY_UP:
      if (moveCamera)
      {
        if (itsChangeRot)
          cvmSet(itsCameraRotation, 0, 0, cvGetReal2D(itsCameraRotation, 0, 0) + M_PI/180);
        else
          cvmSet(itsCameraTranslation, 0, 0, cvGetReal2D(itsCameraTranslation, 0, 0) + 1);
      } else {
        itsCameraCtrl->movePanTilt(1, 0, true); //move relative
      }

      break;
    case KEY_DOWN:
      if (moveCamera)
      {
        if (itsChangeRot)
          cvmSet(itsCameraRotation, 0, 0, cvGetReal2D(itsCameraRotation, 0, 0) - M_PI/180);
        else
          cvmSet(itsCameraTranslation, 0, 0, cvGetReal2D(itsCameraTranslation, 0, 0) - 1);
      } else {
        itsCameraCtrl->movePanTilt(-1, 0, true); //move relative
      }
      break;
    case KEY_LEFT:
      if (moveCamera)
      {
        if (itsChangeRot)
          cvmSet(itsCameraRotation, 0, 1, cvGetReal2D(itsCameraRotation, 0, 1) + M_PI/180);
        else
          cvmSet(itsCameraTranslation, 0, 1, cvGetReal2D(itsCameraTranslation, 0, 1) + 1);
      } else {
        itsCameraCtrl->movePanTilt(0, 1, true); //move relative
      }
      break;
    case KEY_RIGHT:
      if (moveCamera)
      {
        if (itsChangeRot)
          cvmSet(itsCameraRotation, 0, 1, cvGetReal2D(itsCameraRotation, 0, 1) - M_PI/180);
        else
          cvmSet(itsCameraTranslation, 0, 1, cvGetReal2D(itsCameraTranslation, 0, 1) - 1);
      } else {
        itsCameraCtrl->movePanTilt(0, -1, true); //move relative
      }
      break;
    case 38: //a
      if (moveCamera)
      {
        if (itsChangeRot)
          cvmSet(itsCameraRotation, 0, 2, cvGetReal2D(itsCameraRotation, 0, 2) + M_PI/180);
        else
          cvmSet(itsCameraTranslation, 0, 2, cvGetReal2D(itsCameraTranslation, 0, 2) + 1);
      } else {
        itsCameraCtrl->zoom(1, true); //move relative
      }
      break;
    case 52: //z
      if (moveCamera)
      {
        if (itsChangeRot)
          cvmSet(itsCameraRotation, 0, 2, cvGetReal2D(itsCameraRotation, 0, 2) - M_PI/180);
        else
          cvmSet(itsCameraTranslation, 0, 2, cvGetReal2D(itsCameraTranslation, 0, 2) - 1);
      } else {
        itsCameraCtrl->zoom(-1, true); //move relative
      }
      break;
    case 39: //s
      {
        //itsSaveCorners = true;
        //itsZoomOut=true;
        itsCameraCtrl->zoom(+10, true); //move relative

        //Point2D<int> target(-1,-1);
        //q.post(rutz::make_shared(new SimEventSetVisualTracker(this, target)));

      }

      break;
    case 54: //c
      calibrate(itsCorners);
      itsCurrentObjName = "";
      break;
    case 42: //g
      itsDrawGrid = !itsDrawGrid;
      {
        double d0 = cvGetReal2D( itsDistortionCoeffs, 0, 0);
        double d1 = cvGetReal2D( itsDistortionCoeffs, 1, 0);
        double d2 = cvGetReal2D( itsDistortionCoeffs, 2, 0);
        double d3 = cvGetReal2D( itsDistortionCoeffs, 3, 0);
        printf( "distortion_coeffs ( %7.7lf, %7.7lf, %7.7lf, %7.7lf)\n", d0, d1, d2, d3);

        double ir00 = cvGetReal2D( itsIntrinsicMatrix, 0, 0);
        double ir01 = cvGetReal2D( itsIntrinsicMatrix, 0, 1);
        double ir02 = cvGetReal2D( itsIntrinsicMatrix, 0, 2);
        double ir10 = cvGetReal2D( itsIntrinsicMatrix, 1, 0);
        double ir11 = cvGetReal2D( itsIntrinsicMatrix, 1, 1);
        double ir12 = cvGetReal2D( itsIntrinsicMatrix, 1, 2);
        double ir20 = cvGetReal2D( itsIntrinsicMatrix, 2, 0);
        double ir21 = cvGetReal2D( itsIntrinsicMatrix, 2, 1);
        double ir22 = cvGetReal2D( itsIntrinsicMatrix, 2, 2);
        printf( "intrinsics ( %7.5lf, %7.5lf, %7.5lf)\n", ir00, ir01, ir02);
        printf( "           ( %7.5lf, %7.5lf, %7.5lf)\n", ir10, ir11, ir12);
        printf( "           ( %7.5lf, %7.5lf, %7.5lf)\n", ir20, ir21, ir22);

        printf( "Rotation: %f %f %f\n",
            cvGetReal2D(itsCameraRotation, 0, 0),
            cvGetReal2D(itsCameraRotation, 0, 1),
            cvGetReal2D(itsCameraRotation, 0, 2));
        printf( "Translation: %f %f %f\n",
            cvGetReal2D(itsCameraTranslation, 0, 0),
            cvGetReal2D(itsCameraTranslation, 0, 1),
            cvGetReal2D(itsCameraTranslation, 0, 2));
      }
      break;
    case 27: //r
      itsChangeRot = !itsChangeRot;
      //recognize
      //LINFO("What is this object");
      //std::getline(std::cin, itsCurrentObjName);
      //itsZoomOut=true;

      break;

  }

  //cvmSet(itsIntrinsicMatrix, 0, 0, 577.94189); cvmSet(itsIntrinsicMatrix, 0, 1, 0); cvmSet(itsIntrinsicMatrix, 0, 2, 159.500);
  //cvmSet(itsIntrinsicMatrix, 1, 0, 0); cvmSet(itsIntrinsicMatrix, 1, 1, 2235823.50000 ); cvmSet(itsIntrinsicMatrix, 1, 2, 119.5);

}


void PCamera::calibrate(std::vector<CvPoint2D32f>& corners)
{
  int rows = 7;
  int cols = 7;

  int nViews = corners.size()/(rows*cols);
  LINFO("Calibrate: %i views", nViews);

  if (nViews <= 0)
  {
    LINFO("No corners avl");
    return;
  }
  float gridSize = 40;

  // Set up the object points matrix
  // Squares are size set in defines found in header file.
  CvMat* object_points = cvCreateMat(corners.size(), 3, CV_32FC1);
  for(uint k=0; k < corners.size()/(rows*cols); k++ )
  {
    for(int i=0; i < cols; i++ )
    {
      for(int j=0; j < rows; j++ )
      {
        cvmSet( object_points, k*(rows*cols) + i*rows + j, 0, gridSize*j );        // x coordinate (35mm size of square)
        cvmSet( object_points, k*(rows*cols) + i*rows + j, 1, gridSize*i ); // y coordinate
        cvmSet( object_points, k*(rows*cols) + i*rows + j, 2, 0 ); // z coordinate
      }
    }
  }

  //for (uint j = 0; j < corners.size(); j++){
  //  cvSetReal2D( object_points, j, 0, ( j % rows) * gridSize ); //0.4m
  //  cvSetReal2D( object_points, j, 1, ( j / rows) * gridSize );
  //  cvSetReal2D( object_points, j, 2, 0.0 );
        //}

        // Set up the matrix of points per image
        CvMat* point_counts = cvCreateMat(1, nViews, CV_32SC1);
        for(int i=0; i < nViews; i++ )
                cvSetReal1D( point_counts, i, rows*cols );

        // Copy corners found to matrix
        CvMat image_points = cvMat(corners.size(), 2, CV_32FC1, &corners[0]);



  int flags = 0;
  CvMat* cameraRotation = cvCreateMat( nViews, 3, CV_64FC1);
  CvMat* cameraTranslation = cvCreateMat( nViews, 3, CV_64FC1);

  //flags = CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_USE_INTRINSIC_GUESS;
  flags =  CV_CALIB_USE_INTRINSIC_GUESS;
  cvCalibrateCamera2( object_points, &image_points, point_counts, cvSize(320,240),
      itsIntrinsicMatrix, itsDistortionCoeffs,
      cameraRotation,
      cameraTranslation,
      flags);


  //display results
 // Image<byte> in = itsPCameraCellsInput[0];
 // Image<byte> out(in.getDims(), ZEROS);

 // cvUndistort2( img2ipl(in), img2ipl(out), itsIntrinsicMatrix, itsDistortionCoeffs);

 // //itsDebugImg = out;



  cvReleaseMat( &object_points);
  cvReleaseMat( &point_counts);

}

void PCamera::displayExtrinsic(std::vector<CvPoint2D32f>& corners)
{

  int rows = 7;
  //int cols = 6;

  Image<byte> tmp = itsPCameraCellsInput[0];

  float gridSize = 40;
  //Get
  CvMat *image_points_ex = cvCreateMat( corners.size(), 2, CV_64FC1);

  for (uint j = 0; j < corners.size(); j++){
    cvSetReal2D( image_points_ex, j, 0, corners[j].x);
    cvSetReal2D( image_points_ex, j, 1, corners[j].y);
  }

  //int views = 1;

  CvMat *object_points_ex = cvCreateMat( corners.size(), 3, CV_64FC1);
  for (uint j = 0; j < corners.size(); j++){
                cvSetReal2D( object_points_ex, j, 0, ( j % rows) * gridSize ); //0.4m
                cvSetReal2D( object_points_ex, j, 1, ( j / rows) * gridSize );
                cvSetReal2D( object_points_ex, j, 2, 0.0 );
        }

  //cvSetReal2D( itsCameraTranslation, 0, 2, 782.319961 );
  cvFindExtrinsicCameraParams2( object_points_ex,
      image_points_ex,
      itsIntrinsicMatrix,
      itsDistortionCoeffs,
      itsCameraRotation,
      itsCameraTranslation);

  //cvmSet(itsCameraRotation, 0, 0, -1.570442 );
  //cvmSet(itsCameraRotation, 0, 1, 0.028913 );
  //cvmSet(itsCameraRotation, 0, 2, 0.028911 );

  //cvmSet(itsCameraTranslation, 0, 0, -57.067339 );
  //cvmSet(itsCameraTranslation, 0, 1, 0.005114 );
  //cvmSet(itsCameraTranslation, 0, 2, 777.677687 );


  //printf( "Rotation2: %f %f %f\n",
  //    cvGetReal2D(itsCameraRotation, 0, 0),
  //    cvGetReal2D(itsCameraRotation, 0, 1),
  //    cvGetReal2D(itsCameraRotation, 0, 2));
  //printf( "Translation2: %f %f %f\n",
  //    cvGetReal2D(itsCameraTranslation, 0, 0),
  //    cvGetReal2D(itsCameraTranslation, 0, 1),
  //    cvGetReal2D(itsCameraTranslation, 0, 2));

  CvMat *rot_mat = cvCreateMat( 3, 3, CV_64FC1);
  cvRodrigues2( itsCameraRotation, rot_mat, 0);

  int  NUM_GRID         = 12; //21
  CvMat *my_3d_point = cvCreateMat( 3, NUM_GRID * NUM_GRID + 1, CV_64FC1);
        CvMat *my_image_point = cvCreateMat( 2, NUM_GRID * NUM_GRID + 1, CV_64FC1);

  for ( int i = 0; i < NUM_GRID; i++){
                for ( int j = 0; j < NUM_GRID; j++){
                        cvSetReal2D( my_3d_point, 0, i * NUM_GRID + j, -(gridSize*4) + (i * gridSize));
                        cvSetReal2D( my_3d_point, 1, i * NUM_GRID + j, -(gridSize*4) + (j * gridSize));
                        cvSetReal2D( my_3d_point, 2, i * NUM_GRID + j, 0.0);
                }
        }

  cvSetReal2D( my_3d_point, 0, NUM_GRID*NUM_GRID, 0.0);
        cvSetReal2D( my_3d_point, 1, NUM_GRID*NUM_GRID, 0.0);
        cvSetReal2D( my_3d_point, 2, NUM_GRID*NUM_GRID, 15);



  cvProjectPoints2( my_3d_point,
                                        itsCameraRotation,
                                        itsCameraTranslation,
                                        itsIntrinsicMatrix,
                                        itsDistortionCoeffs,
                                        my_image_point);

  for ( int i = 0; i < NUM_GRID; i++){
                for ( int j = 0; j < NUM_GRID-1; j++){
                        int im_x1 = (int)cvGetReal2D( my_image_point, 0, i * NUM_GRID + j);
                        int im_y1 = (int)cvGetReal2D( my_image_point, 1, i * NUM_GRID + j);
                        int im_x2 = (int)cvGetReal2D( my_image_point, 0, i * NUM_GRID + j+1);
                        int im_y2 = (int)cvGetReal2D( my_image_point, 1, i * NUM_GRID + j+1);

                        cvLine( img2ipl(tmp), cvPoint( im_x1, im_y1), cvPoint( im_x2, im_y2), CV_RGB( 0, 255, 0), 1);
                }
        }
        for ( int j = 0; j < NUM_GRID; j++){
                for ( int i = 0; i < NUM_GRID-1; i++){
                        int im_x1 = (int)cvGetReal2D( my_image_point, 0, i * NUM_GRID + j);
                        int im_y1 = (int)cvGetReal2D( my_image_point, 1, i * NUM_GRID + j);
                        int im_x2 = (int)cvGetReal2D( my_image_point, 0, (i+1) * NUM_GRID + j);
                        int im_y2 = (int)cvGetReal2D( my_image_point, 1, (i+1) * NUM_GRID + j);

                        cvLine( img2ipl(tmp), cvPoint( im_x1, im_y1), cvPoint( im_x2, im_y2), CV_RGB( 0, 255, 0), 1);
                }
        }

        int im_x0 = (int)cvGetReal2D( my_image_point, 0, 0);
        int im_y0 = (int)cvGetReal2D( my_image_point, 1, 0);
        int im_x = (int)cvGetReal2D( my_image_point, 0, NUM_GRID*NUM_GRID);
        int im_y = (int)cvGetReal2D( my_image_point, 1, NUM_GRID*NUM_GRID);
        cvLine( img2ipl(tmp), cvPoint( im_x0, im_y0), cvPoint( im_x, im_y), CV_RGB( 255, 0, 0), 2); //Z axis


  itsDebugImg = tmp;

        cvReleaseMat( &my_3d_point);
        cvReleaseMat( &my_image_point);
        //cvReleaseMat( &image_points_ex);
  //cvReleaseMat( &object_points_ex);
  cvReleaseMat( &rot_mat);

}


void PCamera::projectGrid()
{

  Image<byte> tmp = itsDebugImg; //itsPCameraCellsInput[0];


  float gridSize = 40; //38; // in mm
  CvMat *rot_mat = cvCreateMat( 3, 3, CV_64FC1);
  cvRodrigues2( itsCameraRotation, rot_mat, 0);

  int  NUM_GRID         = 9; //21
  CvMat *my_3d_point = cvCreateMat( 3, NUM_GRID * NUM_GRID + 2, CV_64FC1);
        CvMat *my_image_point = cvCreateMat( 2, NUM_GRID * NUM_GRID + 2, CV_64FC1);

  for ( int i = 0; i < NUM_GRID; i++){
                for ( int j = 0; j < NUM_GRID; j++){
                        cvSetReal2D( my_3d_point, 0, i * NUM_GRID + j, -(gridSize*NUM_GRID/2) + (i * gridSize));
                        cvSetReal2D( my_3d_point, 1, i * NUM_GRID + j, -(gridSize*NUM_GRID/2) + (j * gridSize));
                        cvSetReal2D( my_3d_point, 2, i * NUM_GRID + j, 0.0);
                }
        }

  cvSetReal2D( my_3d_point, 0, NUM_GRID*NUM_GRID, 0);
        cvSetReal2D( my_3d_point, 1, NUM_GRID*NUM_GRID, 0);
        cvSetReal2D( my_3d_point, 2, NUM_GRID*NUM_GRID, 0);

  cvSetReal2D( my_3d_point, 0, NUM_GRID*NUM_GRID+1, 0);
        cvSetReal2D( my_3d_point, 1, NUM_GRID*NUM_GRID+1, 0);
        cvSetReal2D( my_3d_point, 2, NUM_GRID*NUM_GRID+1, 30);



  cvProjectPoints2( my_3d_point,
                                        itsCameraRotation,
                                        itsCameraTranslation,
                                        itsIntrinsicMatrix,
                                        itsDistortionCoeffs,
                                        my_image_point);

  for ( int i = 0; i < NUM_GRID; i++){
                for ( int j = 0; j < NUM_GRID-1; j++){
                        int im_x1 = (int)cvGetReal2D( my_image_point, 0, i * NUM_GRID + j);
                        int im_y1 = (int)cvGetReal2D( my_image_point, 1, i * NUM_GRID + j);
                        int im_x2 = (int)cvGetReal2D( my_image_point, 0, i * NUM_GRID + j+1);
                        int im_y2 = (int)cvGetReal2D( my_image_point, 1, i * NUM_GRID + j+1);

                        cvLine( img2ipl(tmp), cvPoint( im_x1, im_y1), cvPoint( im_x2, im_y2), CV_RGB( 0, 255, 0), 1);
                }
        }
        for ( int j = 0; j < NUM_GRID; j++){
                for ( int i = 0; i < NUM_GRID-1; i++){
                        int im_x1 = (int)cvGetReal2D( my_image_point, 0, i * NUM_GRID + j);
                        int im_y1 = (int)cvGetReal2D( my_image_point, 1, i * NUM_GRID + j);
                        int im_x2 = (int)cvGetReal2D( my_image_point, 0, (i+1) * NUM_GRID + j);
                        int im_y2 = (int)cvGetReal2D( my_image_point, 1, (i+1) * NUM_GRID + j);

                        cvLine( img2ipl(tmp), cvPoint( im_x1, im_y1), cvPoint( im_x2, im_y2), CV_RGB( 0, 255, 0), 1);
                }
        }

        int im_x0 = (int)cvGetReal2D( my_image_point, 0, NUM_GRID*NUM_GRID);
        int im_y0 = (int)cvGetReal2D( my_image_point, 1, NUM_GRID*NUM_GRID);
        int im_x = (int)cvGetReal2D( my_image_point, 0, NUM_GRID*NUM_GRID+1);
        int im_y = (int)cvGetReal2D( my_image_point, 1, NUM_GRID*NUM_GRID+1);
        cvLine( img2ipl(tmp), cvPoint( im_x0, im_y0), cvPoint( im_x, im_y), CV_RGB( 255, 0, 0), 2); //Z axis


  itsDebugImg = tmp;

        cvReleaseMat( &my_3d_point);
        cvReleaseMat( &my_image_point);
  cvReleaseMat( &rot_mat);




}

void PCamera::projectPoints(std::vector<Point3D<float> >& points)
{


  Image<byte> tmp = itsDebugImg;
  //Image<byte> tmp = itsPCameraCellsInput[0];


  CvMat *rot_mat = cvCreateMat( 3, 3, CV_64FC1);
  cvRodrigues2( itsCameraRotation, rot_mat, 0);

  //Project qube
  CvMat *my_3d_point = cvCreateMat( 3, points.size(), CV_64FC1);
        CvMat *my_image_point = cvCreateMat( 2, points.size(), CV_64FC1);

  for(uint i=0; i<points.size(); i++)
  {
    cvSetReal2D( my_3d_point, 0, i, points[i].x-20);
    cvSetReal2D( my_3d_point, 1, i, points[i].y-20);
    cvSetReal2D( my_3d_point, 2, i, points[i].z);
        }

  cvProjectPoints2( my_3d_point,
                                        itsCameraRotation,
                                        itsCameraTranslation,
                                        itsIntrinsicMatrix,
                                        itsDistortionCoeffs,
                                        my_image_point);

  for(uint i=0; i<points.size(); i++)
  {
                        int im_x1 = (int)cvGetReal2D( my_image_point, 0, i );
                        int im_y1 = (int)cvGetReal2D( my_image_point, 1, i );

                        int im_x2 = (int)cvGetReal2D( my_image_point, 0, (i+1)%points.size() );
                        int im_y2 = (int)cvGetReal2D( my_image_point, 1, (i+1)%points.size() );

      drawLine(tmp, Point2D<int>(im_x1, im_y1), Point2D<int>(im_x2, im_y2),
          (byte)0, 3);
  }

//  for ( int i = 0; i < NUM_GRID; i++){
//                for ( int j = 0; j < NUM_GRID-1; j++){
//                        int im_x1 = (int)cvGetReal2D( my_image_point, 0, i * NUM_GRID + j);
//                        int im_y1 = (int)cvGetReal2D( my_image_point, 1, i * NUM_GRID + j);
//                        int im_x2 = (int)cvGetReal2D( my_image_point, 0, i * NUM_GRID + j+1);
//                        int im_y2 = (int)cvGetReal2D( my_image_point, 1, i * NUM_GRID + j+1);
//
//                        cvLine( img2ipl(tmp), cvPoint( im_x1, im_y1), cvPoint( im_x2, im_y2), CV_RGB( 0, 255, 0), 1);
//                }
//        }
//        for ( int j = 0; j < NUM_GRID; j++){
//                for ( int i = 0; i < NUM_GRID-1; i++){
//                        int im_x1 = (int)cvGetReal2D( my_image_point, 0, i * NUM_GRID + j);
//                        int im_y1 = (int)cvGetReal2D( my_image_point, 1, i * NUM_GRID + j);
//                        int im_x2 = (int)cvGetReal2D( my_image_point, 0, (i+1) * NUM_GRID + j);
//                        int im_y2 = (int)cvGetReal2D( my_image_point, 1, (i+1) * NUM_GRID + j);
//
//                        cvLine( img2ipl(tmp), cvPoint( im_x1, im_y1), cvPoint( im_x2, im_y2), CV_RGB( 0, 255, 0), 1);
//                }
//        }
//
//        int im_x0 = (int)cvGetReal2D( my_image_point, 0, 0);
//        int im_y0 = (int)cvGetReal2D( my_image_point, 1, 0);
//        int im_x = (int)cvGetReal2D( my_image_point, 0, NUM_GRID*NUM_GRID);
//        int im_y = (int)cvGetReal2D( my_image_point, 1, NUM_GRID*NUM_GRID);
//        cvLine( img2ipl(tmp), cvPoint( im_x0, im_y0), cvPoint( im_x, im_y), CV_RGB( 255, 0, 0), 2); //Z axis


  itsDebugImg = tmp;

        cvReleaseMat( &my_3d_point);
        cvReleaseMat( &my_image_point);
  cvReleaseMat( &rot_mat);

}




// ######################################################################
void PCamera::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "PCamera", FrameInfo("PCamera", SRC_POS));
    }
}


Layout<PixRGB<byte> > PCamera::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;

  Image<PixRGB<byte> > tmp = itsCurrentImg;
  for(uint i=0; i<tmp.size(); i++)
    if (itsRenderImg[i] != PixRGB<byte>(0,0,0))
      tmp[i] = itsRenderImg[i];

  for(uint i=0; i<itsSquares.size(); i++)
  {
    for(uint j=0; j<4; j++)
      drawLine(itsCurrentImg,
          itsSquares[i].p[j],
          itsSquares[i].p[(j+1)%4],
          PixRGB<byte>(255,0,0));
  }
  outDisp = hcat(itsCurrentImg, tmp);

  return outDisp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

