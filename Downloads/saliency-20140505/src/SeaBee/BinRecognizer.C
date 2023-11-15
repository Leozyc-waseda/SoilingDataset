/*!@file SeaBee/BinRecognizer.C find bin     */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/BinRecognizer.C $
// $Id: BinRecognizer.C 10794 2009-02-08 06:21:09Z itti $

#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Image/DrawOps.H"

#include "MBARI/Geometry2D.H"
#include "Image/OpenCVUtil.H"
#include "Image/ColorOps.H"

#include "SeaBee/VisionRecognizer.H"
#include "SeaBee/BinRecognizer.H"

// ######################################################################
BinRecognizer::BinRecognizer()
{
}

// ######################################################################
BinRecognizer::~BinRecognizer()
{ }

// ######################################################################
uint BinRecognizer::getBinLocation(rutz::shared_ptr<Image<PixRGB <byte> > > image,
                                   rutz::shared_ptr<Image<PixRGB <byte> > > outputImage,
                                   BinRecognizer::BinRecognizeMethod method,
                                   rutz::shared_ptr<Point2D<int> > binCenter,
                                   uint &staleCount
                                   )
{
  CvSeq* squares;
   switch(method)
     {
     case BinRecognizer::CONTOUR:
       squares = getContours(img2ipl(*image));
       getBinCenter(staleCount, binCenter, squares);
       *outputImage = drawSquares(img2ipl(*image), squares);
       return 0;
       break;
     default:
       LERROR("Invalid bin recognizer method specified");
       return 0;
     }
}

void BinRecognizer::getBinCenter(uint &staleCount,
                                 rutz::shared_ptr<Point2D<int> > binCenter,
                                 CvSeq* squares)
{
  CvSeqReader reader;
  int i;

  // initialize reader of the sequence
  cvStartReadSeq( squares, &reader, 0 );

  int x = 0;
  int y = 0;

  // read 4 sequence elements at a time (all vertices of a square)
  for( i = 0; i < squares->total; i += 4 )
    {
      CvPoint pt[4];//, *rect = pt;
      //int count = 4;

      // read 4 vertices
      CV_READ_SEQ_ELEM( pt[0], reader );
      CV_READ_SEQ_ELEM( pt[1], reader );
      CV_READ_SEQ_ELEM( pt[2], reader );
      CV_READ_SEQ_ELEM( pt[3], reader );

      x += pt[0].x;
      x += pt[1].x;
      x += pt[2].x;
      x += pt[3].x;

      y += pt[0].y;
      y += pt[1].y;
      y += pt[2].y;
      y += pt[3].y;

    }

  //LINFO("x:%d y:%d",x,y);
  int avgX = x/4;
  int avgY = y/4;

  if(x > 0  && y > 0 && avgY < 100 )
    {
      binCenter->i = avgX;
      binCenter->j = avgY;
      staleCount = 0;
    }
  else
    {
      staleCount++;
    }
}
// ######################################################################
// the function draws all the squares in the image
Image<PixRGB<byte> > BinRecognizer::drawSquares( IplImage* in, CvSeq* squares )
{
  CvSeqReader reader;
  int i;

  // initialize reader of the sequence
  cvStartReadSeq( squares, &reader, 0 );

  // read 4 sequence elements at a time (all vertices of a square)
  for( i = 0; i < squares->total; i += 4 )
    {
      CvPoint pt[4], *rect = pt;
      int count = 4;

      // read 4 vertices
      CV_READ_SEQ_ELEM( pt[0], reader );
      CV_READ_SEQ_ELEM( pt[1], reader );
      CV_READ_SEQ_ELEM( pt[2], reader );
      CV_READ_SEQ_ELEM( pt[3], reader );

      // draw the square as a closed polyline
      cvPolyLine( in, &rect, &count, 1, 1, CV_RGB(0,255,0), 3, CV_AA, 0 );
    }

  return ipl2rgb(in);

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
