/*!@file SeaBee/VisionRecognizer.C */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Kevin Greene <kgreene@usc.edu> & Josh Villbrandt <josh.villbrandt@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/VisionRecognizer.C $
// $Id: VisionRecognizer.C 12962 2010-03-06 02:13:53Z irock $

#ifdef HAVE_OPENCV

#include "SeaBee/VisionRecognizer.H"

// ######################################################################

VisionRecognizer::VisionRecognizer()
{
  //  itsDispWin.reset(new XWinManaged(Dims(320*2,240*2), 0, 0, "Contour Recognizer Display"));
}

std::vector<LineSegment2D> VisionRecognizer::getHoughLines( IplImage cannyImage )
{
#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed in order to use this function");
#else
  // Storage for use in hough transform.
  CvMemStorage* storage = cvCreateMemStorage(0);

  // Perform hough transform and store hough lines.
  CvSeq* cvLines = cvHoughLines2(&cannyImage, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 30, 20, 10);


  // Storage for hough line segments.
  std::vector <LineSegment2D> lineSegments;

  // Loop through hough lines, store them as line segments, and draw lines in output image.
  for(int i = 0; i < cvLines->total; i++ )
  {
    // Get a line.
    CvPoint* line = (CvPoint*)cvGetSeqElem(cvLines,i);

    // Get line end points.
    Point2D<int> pt1 = Point2D<int>(line[0].x,line[0].y);
    Point2D<int> pt2 = Point2D<int>(line[1].x,line[1].y);

    // Create line segment from endpoints and store.
    lineSegments.push_back(LineSegment2D(pt1,pt2));
  }
  cvReleaseMemStorage( &storage );

  return lineSegments;
#endif // HAVE_OPENCV
}

// ######################################################################

std::vector<LineSegment2D> VisionRecognizer::pruneHoughLines (const std::vector<LineSegment2D> lineSegments)
{
  uint numLines = lineSegments.size();
  if(numLines == 0) { LDEBUG("No hough lines to prune"); }

  std::vector< std::vector<LineSegment2D> > pipeLines;

  //Go through all the lines
  for(uint r = 0; r < numLines; r++)
    {
      int lnIndex = -1;

      //check to see if the current lines fits into a bucket
      for(uint c = 0; c < pipeLines.size(); c++)
        {
          LineSegment2D pipeLine = pipeLines[c][0];

          if(pipeLine.isValid() && lineSegments[r].angleBetween(pipeLine) < 5*(M_PI/180))//convert 5 degrees to radians
          {
            lnIndex = c;
            break;
          }
        }

      //if the line fits into a pre-existing bucket, add it to the bucket
      if( lnIndex > 0 )
        {
          pipeLines[lnIndex].push_back(lineSegments[r]);
          //average the old bucket's value with the new line added
          //so as to create a moving bucket
          Point2D<int> newPt1 =
            Point2D<int>(((lineSegments[r].point1().i + pipeLines[lnIndex][0].point1().i)/2),
                         ((lineSegments[r].point1().j + pipeLines[lnIndex][0].point1().j)/2));

          Point2D<int> newPt2 = Point2D<int>(((lineSegments[r].point2().i + pipeLines[lnIndex][0].point2().i)/2),
                            ((lineSegments[r].point2().j + pipeLines[lnIndex][0].point2().j)/2));

          pipeLines[lnIndex][0] = LineSegment2D(newPt1,newPt2);

        }
      //otherwise, create a new bucket
      else
        {
          std::vector<LineSegment2D> newCntrLines;
          newCntrLines.push_back(lineSegments[r]);
          pipeLines.push_back(newCntrLines);
        }
    }

  std::vector<LineSegment2D> centerPipeLines;

  uint pipeLineSize = pipeLines.size();

  for(uint c = 0; c < pipeLineSize; c++)
    {
      centerPipeLines.push_back(pipeLines[c][0]);
    }
//  std::vector<LineSegment2D> centerPipeLines;

//   Point2D<int> two = Point2D<int>(2,2);

//   for(uint c = 0; c < pipeLines.size(); c++)
//     {
//       if(pipeLines[c].size() == 2)
//         {
//           Point2D<int> endPoint1 = Point2D<int>((pipeLines[c][0].point1()+pipeLines[c][1].point1())/two);
//           Point2D<int> endPoint2 = Point2D<int>((pipeLines[c][0].point2()+pipeLines[c][1].point2())/two);

//           centerPipeLines.push_back(LineSegment2D(endPoint1,endPoint2));
//         }
//     }

  return centerPipeLines;
}
// ######################################################################

CvSeq* VisionRecognizer::getContours( IplImage* img )
{

  Image< PixRGB<byte> > dispImg(320*2,240*2, ZEROS);

  int thresh = 50;

  CvMemStorage* storage = 0;

  // create memory storage that will contain all the dynamic data
  storage = cvCreateMemStorage(0);

  CvSeq* contours;
  int i, c, l, N = 11;
  CvSize sz = cvSize( img->width & -2, img->height & -2 );
  IplImage* timg = cvCloneImage( img ); // make a copy of input image
  IplImage* gray = cvCreateImage( sz, 8, 1 );
  IplImage* pyr = cvCreateImage( cvSize(sz.width/2, sz.height/2), 8, 3 );
  IplImage* tgray;
  CvSeq* result;
  double s, t;
  // create empty sequence that will contain points -
  // 4 points per square (the square's vertices)
  CvSeq* squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );

  // select the maximum ROI in the image
  // with the width and height divisible by 2
  cvSetImageROI( timg, cvRect( 0, 0, sz.width, sz.height ));

  // down-scale and upscale the image to filter out the noise
  cvPyrDown( timg, pyr, 7 );
  cvPyrUp( pyr, timg, 7 );
  tgray = cvCreateImage( sz, 8, 1 );

  // find squares in every color plane of the image
  for( c = 0; c < 3; c++ )
    {
      // extract the c-th color plane
      cvSetImageCOI( timg, c+1 );
      cvCopy( timg, tgray, 0 );

      // try several threshold levels
      for( l = 0; l < N; l++ )
        {
          // hack: use Canny instead of zero threshold level.
          // Canny helps to catch squares with gradient shading
          if( l == 0 )
            {
              // apply Canny. Take the upper threshold from slider
              // and set the lower to 0 (which forces edges merging)
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
          cvFindContours( gray, storage, &contours, sizeof(CvContour),
                          CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );

          // test each contour
          while( contours )
            {
              // approximate contour with accuracy proportional
              // to the contour perimeter
              result = cvApproxPoly( contours, sizeof(CvContour), storage,
                                     CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );
              // square contours should have 4 vertices after approximation
              // relatively large area (to filter out noisy contours)
              // and be convex.
              // Note: absolute value of an area is used because
              // area may be positive or negative - in accordance with the
              // contour orientation
              if( result->total == 4 &&
                  fabs(cvContourArea(result,CV_WHOLE_SEQ)) > 1000 &&
                  fabs(cvContourArea(result,CV_WHOLE_SEQ)) < 2500 &&
                  cvCheckContourConvexity(result) )
                {
                  s = 0;

                  //LINFO("AREA: %d",int(fabs(cvContourArea(result,CV_WHOLE_SEQ))));

                  for( i = 0; i < 5; i++ )
                    {
                      // find minimum angle between joint
                      // edges (maximum of cosine)
                      if( i >= 2 )
                        {
                          t = fabs(angle(
                                         (CvPoint*)cvGetSeqElem( result, i ),
                                         (CvPoint*)cvGetSeqElem( result, i-2 ),
                                         (CvPoint*)cvGetSeqElem( result, i-1 )));
                          s = s > t ? s : t;
                        }
                    }

                  // if cosines of all angles are small
                  // (all angles are ~90 degree) then write quandrange
                  // vertices to resultant sequence
                  if( s < 0.3 )
                    for( i = 0; i < 4; i++ )
                      cvSeqPush( squares,
                                 (CvPoint*)cvGetSeqElem( result, i ));
                }

              // take the next contour
              contours = contours->h_next;
            }
        }
    }


  inplacePaste(dispImg, toRGB(ipl2gray(gray)), Point2D<int>(0,0));
  inplacePaste(dispImg, ipl2rgb(pyr), Point2D<int>(320,0));
  inplacePaste(dispImg, toRGB(ipl2gray(tgray)), Point2D<int>(0,240));
  inplacePaste(dispImg, ipl2rgb(timg), Point2D<int>(320,240));

  //  itsDispWin->drawImage(dispImg, 0, 0);

  // release all the temporary images
  cvReleaseImage( &gray );
  cvReleaseImage( &pyr );
  cvReleaseImage( &tgray );
  cvReleaseImage( &timg );

  return squares;
}

      // ######################################################################

IplImage VisionRecognizer::getCannyImage( Image<byte> colorSegmentedImage )
{
#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed in order to use this function");
#else
        // Find edges of segmented image using canny.
  IplImage *edge = cvCreateImage( cvGetSize( img2ipl( colorSegmentedImage ) ), 8, 1 );
  cvCanny( img2ipl( luminance( colorSegmentedImage ) ), edge, 100, 150, 3 );//150,200,3

        return *edge;
#endif // HAVE_OPENCV
}

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double VisionRecognizer::angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 )
{
    double dx1 = pt1->x - pt0->x;
    double dy1 = pt1->y - pt0->y;
    double dx2 = pt2->x - pt0->x;
    double dy2 = pt2->y - pt0->y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
