/*!@file Beobot/test-roadShape.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; filed July 23, 2001, following provisional applications     //
// No. 60/274,674 filed March 8, 2001 and 60/288,724 filed May 4, 2001).//
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/test-roadShape.C $
// $Id: test-roadShape.C 14376 2011-01-11 02:44:34Z pez $
//

//!To Display frames and associated interpretation of MetaData

#include "GUI/XWindow.H"
#include "Image/CutPaste.H"  // for inplacePaste()
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include <cmath>
#include <signal.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

namespace
{
  //! Metadata information (typically obtained from GPS)
  class MetaData
  {
  public:
    //! Constructor
    MetaData();

    //! Constructor
    MetaData( const MetaData& m );

    //! Constructor
    MetaData( std::string s );

    //! Parse from string
    void parseMetaData( std::string s );

    //! Access function
    float getAccel() const;

    //! Access function
    float getSteer() const;

    //! Access function
    float getLat() const;

    //! Access function
    float getLon() const;

    //! Access function
    float getEV() const;

    //! Access function
    float getNV() const;

    //protected:
    float accel; //!< Acceleration
    float steer; //!< Steering
    float lat;   //!< Latitude
    float lon;   //!< Longitude
    float ev;    //!< Velocity in east direction
    float nv;    //!< Velocity in north direction
  };

  MetaData::MetaData() :
    accel(0), steer(0), lat(0), lon(0), ev(0), nv(0)
  { }

  MetaData::MetaData( const MetaData& m ) :
    accel( m.accel ), steer( m.steer ), lat( m.lat ), lon( m.lon),
    ev( m.ev ), nv( m.nv )
  { }

  MetaData::MetaData( std::string s ) :
    accel(0), steer(0), lat(0), lon(0), ev(0), nv(0)
  {
    parseMetaData( s );
  }

  void MetaData::parseMetaData( std::string s )
  {
    std::string temp = "";
    int space;
    while( ( space = s.find( " ", 0 ) ) != int(std::string::npos) )
      {
        temp = s.substr( 0, space );
        s = s.substr( space, s.length() - space );

        if( temp.find( "ACCEL:", 0 ) != std::string::npos )
          accel = atof( temp.c_str() );
        if( temp.find("STEER:", 0) != std::string::npos )
          steer = atof( temp.c_str() );
        if( temp.find("LAT:", 0 ) != std::string::npos )
          lat = atof( temp.c_str() );
        if( temp.find("LON:", 0 ) != std::string::npos )
          lon = atof( temp.c_str() );
        if( temp.find("EV:", 0) != std::string::npos )
          ev = atof( temp.c_str() );
        if( temp.find("NV:", 0) != std::string::npos )
          nv = atof( temp.c_str() );
      }
  }

  float MetaData::getAccel() const
  { return accel; }

  float MetaData::getSteer() const
  { return steer; }

  float MetaData::getLat() const
  { return lat; }

  float MetaData::getLon() const
  { return lon; }

  float MetaData::getEV() const
  { return ev; }

  float MetaData::getNV() const
  { return nv; }
}

// ######################################################################
int main(int argc, char **argv)
{
  initRandomNumbers();

  int32 frame;

  int initFrame;
  char imageNameBase[100];

  // command line help
  if( argc >= 2 ) strcpy( imageNameBase, argv[1] );
  else LFATAL("arguments = dir/ [initFrame]");

  if( argc >= 3 ) initFrame = atoi( argv[2] );
  else initFrame = 0;

  Image< PixRGB<byte> > col_image =
    Raster::ReadRGB(sformat("%sframe%06d.ppm", imageNameBase, initFrame));

  //Here we are assuming all of the images will have the same dimensions
  const int HEIGHT = col_image.getHeight();
  const int WIDTH = col_image.getWidth();

  //Points dividing the window into image sized segments [3x3],
  //Based on image size
  Point2D<int> up_left(0, 0);
  Point2D<int> up_mid( WIDTH, 0 );
  //Point2D<int> up_right( WIDTH*2, 0 );
  //Point2D<int> mid_left(0, HEIGHT );
  //Point2D<int> mid_mid( WIDTH, HEIGHT );
  //Point2D<int> mid_right( WIDTH*2, HEIGHT );
  //Point2D<int> low_left(0, HEIGHT*2 );
  //Point2D<int> low_mid( WIDTH, HEIGHT*2 );
  //Point2D<int> low_right( WIDTH*2, HEIGHT*2 );

  //Acceleration representation rectangle
  Rectangle accel;

  //Steering represenation rectangle
  Rectangle steer;


  Image< PixRGB<byte> > final_image( WIDTH*2, HEIGHT, ZEROS );

  // this is to show the final composite image
  XWindow xwindow( Dims(WIDTH*2, HEIGHT*2) );


  // initialization of the beast

  //This image will show the steering and acceleration vector representations
  //Image< PixRGB<byte> > radioImage(  WIDTH, HEIGHT, ZEROS );
  Image< PixRGB<byte> > radioImage = Raster::ReadRGB("speedo.ppm");

  //Image< PixRGB<byte> > clear( WIDTH, HEIGHT, ZEROS );

  MetaData mData;

  // big loop
  for(frame = initFrame; ; frame+=1)
    {
      // read current frame
      col_image = Raster::ReadRGB(sformat("%sframe%06d.ppm",
                                          imageNameBase, frame));
      // and display it for debug

      inplacePaste( final_image, col_image, up_left );

      //now we want to create some sort of representaiton of the speed
      //And direction we are trying to go

      //Read in the MetaData from the image
      mData = MetaData(Raster::getImageComments
                       (sformat("%sframe%06d.ppm",
                                imageNameBase, frame)));

      //Since accel is nicely between -1 and 1, we can simply multiply it
      //by half of the image height to get an offset for drawing our acceleration
      //Rectangle.

      int vert = (int) 0;//(mData.accel * (HEIGHT/2));
      int hori = (int) 0;//(mData.steer * (WIDTH/2));

      LINFO("VERT: %d\nHORI: %d", vert, hori);
      if( vert == 0 )
        vert = 1;
      if( hori == 0 )
        hori = 1;


      if (frame > 100) {
        vert = 30;
        hori = 30;
      }

      //Set the acceleration rectangle
      if( vert < 0 )
        {
          LINFO("uu:%d ll:%d bb:%d rr:%d", HEIGHT + vert, 0, HEIGHT / 2, 10);
          accel = Rectangle::tlbrI( HEIGHT / 2, 0,
                                   HEIGHT / 2 - vert, 10 );
        }
      else
        {
          LINFO("uu:%d ll:%d bb:%d rr:%d", HEIGHT / 2, 0, HEIGHT + vert, 10);
          accel = Rectangle::tlbrI( HEIGHT / 2 - vert, 0,
                                   HEIGHT / 2, 10 );
        }

      //Set the steering rectangle
      if( hori > 0 )
        {
          LINFO("uu:%d ll:%d bb:%d rr:%d", HEIGHT - 10, WIDTH / 2, HEIGHT, WIDTH / 2 + hori);
          steer = Rectangle::tlbrI( HEIGHT - 11, WIDTH / 2,
                                   HEIGHT - 1, WIDTH / 2 + hori );
        }
      else
        {
          LINFO("uu:%d ll:%d bb:%d rr:%d", HEIGHT - 10, WIDTH / 2 + hori,
                HEIGHT, WIDTH / 2 );
          steer = Rectangle::tlbrI( HEIGHT - 11, WIDTH / 2 + hori,
                                   HEIGHT - 1, WIDTH / 2 );
        }

      PixRGB<byte> whitePixel(255,36,255);

      //radioImage.clear( blackPixel );

      drawRect( radioImage, accel, whitePixel, 1 );
      drawRect( radioImage, steer, whitePixel, 1 );

      //Point2D<int> p1( WIDTH / 2, HEIGHT / 2 );
      Point2D<int> p1( WIDTH / 2, HEIGHT );
      Point2D<int> p2( WIDTH / 2 + hori, HEIGHT / 2 - vert );

      drawArrow(radioImage, p1, p2, whitePixel, 2);


      inplacePaste( final_image, radioImage, up_mid );

      //drawImage( xwindow, final_image );

      LINFO("Writing: %smovie/frame%06d.ppm", imageNameBase, frame);
      Raster::WriteRGB( final_image,
                        sformat("%smovie/frame%06d.ppm", imageNameBase, frame) );
    }

  LINFO( "END OF test-MetaData" );
  exit(0);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
