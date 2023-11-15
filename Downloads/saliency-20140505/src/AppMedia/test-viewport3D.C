/*!@file AppMedia/test-viewport3D.C test the opengl viewport */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/test-viewport3D.C $
// $Id: test-viewport3D.C 14486 2011-02-08 05:50:33Z lior $


#include "GUI/ViewPort3D.H"
#include "GUI/SuperQuadric.H"
#include "Util/log.H"
#include "Util/WorkThreadServer.H"
#include "Util/JobWithSemaphore.H"
#include "Component/ModelManager.H"
#include "Raster/GenericFrame.H"
#include "Image/Layout.H"
#include "Image/MatrixOps.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include <stdlib.h>
#include <math.h>


int main(int argc, char *argv[]){

  ModelManager manager("Test Viewport");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);
  // let's get all our ModelComponent instances started:
  manager.start();

  bool testTexture = false;
  bool testFeedback = false;
  bool testSQ = false;
  bool testGetPosition = true;


  //Test texture
  if (testTexture)
  {
    ViewPort3D vp(320,240, false, false, false);
    vp.setCamera(Point3D<float>(0,0,300), Point3D<float>(0,0,0));

    vp.initFrame();
    Image<PixRGB<byte> > texture1(64,64,ZEROS);
    for(int i=0; i<texture1.getHeight(); i++)
      for(int j=0; j<texture1.getWidth(); j++)
      {
        int c = ((((i&0x8)==0)^((j&0x8)==0)))*255;
        texture1.setVal(j,i, PixRGB<byte>(0,c,c));
      }
    uint checkBoard1 = vp.addTexture(texture1);

    for(uint i=0; i<360*2; i++)
    {
      LINFO("i %i", i);

      vp.initFrame();
      vp.bindTexture(checkBoard1);

      glRotatef(i, 0,1,0);
      glBegin(GL_QUADS);


      glTexCoord2f(0.0, 0.0); 
      glVertex3f( 40, 40, 0);                        // Top Right Of The Quad (Front)

      glTexCoord2f(0.0, 1.0); 
      glVertex3f(-40, 40, 0);                        // Top Left Of The Quad (Front)

      glTexCoord2f(1.0, 1.0); 
      glVertex3f(-40,-40, 0);                        // Bottom Left Of The Quad (Front)

      glTexCoord2f(1.0, 0.0); 
      glVertex3f( 40,-40, 0);                        // Bottom Right Of The Quad (Front)
      glEnd();

      Image<PixRGB<byte> > img = flipVertic(vp.getFrame());
      ofs->writeRGB(img, "ViewPort3D", FrameInfo("ViewPort3D", SRC_POS));
      usleep(10000);
    }
  }


  //Check Feedback
  if (testFeedback)
  {
    ViewPort3D vp(320,240, true, true, true);

    vp.setCamera(Point3D<float>(0,0,350), Point3D<float>(0,0,0));

    vp.initFrame();

    glPassThrough (1.0);
    vp.drawBox( Point3D<float>(60,60,0), //Position
        Point3D<float>(0,0,0), //Rotation
        Point3D<float>(30, 30, 50), //size
        PixRGB<byte>(0,256,0)
        );
    glPassThrough (2.0);

    vp.drawCylinder( Point3D<float>(-60,-60,0), //Position
        Point3D<float>(0,0,0), //Rotation
        30, //radius
        50, //length
        PixRGB<byte>(256,256,0)
        );
    

    //Check the feedback
    Image<PixRGB<byte> > tmp(320, 240, ZEROS);
    std::vector<ViewPort3D::Line> lines = vp.getFrameLines();
    for(uint i=0; i<lines.size(); i++)
      drawLine(tmp, Point2D<int>(lines[i].p1), Point2D<int>(lines[i].p2), PixRGB<byte>(255,0,0));
    LINFO("Lines feedback");
    SHOWIMG(tmp);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 
    
  }



  //Normal operation


  if (testGetPosition)
  {
    ViewPort3D vp(320,240, false, false, false);
    vp.setProjectionMatrix(90, 320/240, 0.005, 500);
    vp.setCamera(Point3D<float>(0,0,30), Point3D<float>(-200,0,0));
    while(1)
    {
      vp.initFrame();

      vp.setColor(PixRGB<byte>(255,255,255));
      vp.drawGrid(Dims(300,300), Dims(10,10));
      vp.drawCircle(Point3D<float>(0,0,0),
          Point3D<float>(0,0,0), //Rotation
          1, PixRGB<byte>(255,0,0));
      
      Image<PixRGB<byte> > img = flipVertic(vp.getFrame());

      Point3D<float> loc2D(159,164,1);
      Point3D<float> loc3D = vp.getPosition(loc2D);
      LINFO("2D loc %f,%f,%f => 3D loc %f,%f,%f",
          loc2D.x, loc2D.y, loc2D.z,
          loc3D.x, loc3D.y, loc3D.z);

      loc2D = Point3D<float>(159,23,1);
      loc3D = vp.getPosition(loc2D);
      LINFO("2D loc %f,%f,%f => 3D loc %f,%f,%f",
          loc2D.x, loc2D.y, loc2D.z,
          loc3D.x, loc3D.y, loc3D.z);

      SHOWIMG(img);
      ofs->writeRGB(img, "ViewPort3D", FrameInfo("ViewPort3D", SRC_POS));
    }


    vp.initFrame();

    vp.drawGrid(Dims(300,300), Dims(10,10));
    
    Point3D<float> loc2D(158,120,1);
    Point3D<float> loc3D = vp.getPosition(loc2D);
    LINFO("2D loc %f,%f,%f => 3D loc %f,%f,%f",
        loc2D.x, loc2D.y, loc2D.z,
        loc3D.x, loc3D.y, loc3D.z);

    loc2D = Point3D<float>(213,160,1);
    loc3D = vp.getPosition(loc2D);
    LINFO("2D loc %f,%f,%f => 3D loc %f,%f,%f",
        loc2D.x, loc2D.y, loc2D.z,
        loc3D.x, loc3D.y, loc3D.z);

    //Check the feedback
    Image<PixRGB<byte> > tmp(320, 240, ZEROS);
    std::vector<ViewPort3D::Line> lines = vp.getFrameLines();
    for(uint i=0; i<lines.size(); i++)
      drawLine(tmp, Point2D<int>(lines[i].p1), Point2D<int>(lines[i].p2), PixRGB<byte>(255,0,0));
    LINFO("Lines feedback");
    SHOWIMG(tmp);

  }

  ViewPort3D vp(320,240, false, true, false);
  vp.setCamera(Point3D<float>(0,0,350), Point3D<float>(-180,0,0));


  if (testSQ)
  {
      SuperQuadric superQuadric;
      while(1)
      {
        for(float n=0; n<1; n+=0.5)
          for(float e=0; e<1; e+=0.5)
          {
            for(float rot=0; rot<360; rot++)
            {
              vp.initFrame();

              vp.setColor(PixRGB<byte>(255,255,255));
              glRotatef(rot, 0,0,1);
              glRotatef(45, 1,0,0);


              superQuadric.its_a1 =30; 
              superQuadric.its_a2 = 30;
              superQuadric.its_a3 = 30;
              superQuadric.its_n = n;
              superQuadric.its_e = e;
              superQuadric.its_u1 = -M_PI / 2;
              superQuadric.its_u2 = M_PI / 2;
              superQuadric.its_v1 = -M_PI;
              superQuadric.its_v2 = M_PI;
              superQuadric.its_s1 = 0.0f;
              superQuadric.its_t1 = 0.0f;
              superQuadric.its_s2 = 1.0f;
              superQuadric.its_t2 = 1.0f;

              //glDisable (GL_CULL_FACE);

              superQuadric.solidEllipsoid();

              Image<PixRGB<byte> > img = flipVertic(vp.getFrame());
              ofs->writeRGB(img, "ViewPort3D", FrameInfo("ViewPort3D", SRC_POS));
              usleep(10000);
            }
          }
      }
  }

  int rot = 0;
  while(1)
  {
    vp.initFrame();

    rot = ((rot +1)%360);

    //vp.drawGround(Point2D<float>(100,100),
    //              PixRGB<byte>(255,255,255));

    ////////Test Contour display (counter needs to be specified counter clockwise
    std::vector<Point2D<float> > contour;

    float scale = 5;
    contour.push_back(Point2D<float>(0, 3)*scale);
    contour.push_back(Point2D<float>(1, 1)*scale);
    contour.push_back(Point2D<float>(5, 1)*scale);
    contour.push_back(Point2D<float>(8, 4)*scale);
    contour.push_back(Point2D<float>(10, 4)*scale);
    contour.push_back(Point2D<float>(11, 5)*scale);
    contour.push_back(Point2D<float>(11, 11.5)*scale);
    contour.push_back(Point2D<float>(13, 12)*scale);
    contour.push_back(Point2D<float>(13, 13)*scale);
    contour.push_back(Point2D<float>(10, 13.5)*scale);
    contour.push_back(Point2D<float>(13, 14)*scale);
    contour.push_back(Point2D<float>(13, 15)*scale);
    contour.push_back(Point2D<float>(11, 16)*scale);
    contour.push_back(Point2D<float>(8, 16)*scale);
    contour.push_back(Point2D<float>(7, 15)*scale);
    contour.push_back(Point2D<float>(7, 13)*scale);
    contour.push_back(Point2D<float>(8, 12)*scale);
    contour.push_back(Point2D<float>(7, 11)*scale);
    contour.push_back(Point2D<float>(6, 6)*scale);
    contour.push_back(Point2D<float>(4, 3)*scale);
    contour.push_back(Point2D<float>(3, 2)*scale);
    contour.push_back(Point2D<float>(1, 2)*scale);

    //Center the contour
    Point2D<float> center = centroid(contour);
    for(uint i=0; i<contour.size(); i++)
      contour[i] -= center;

    vp.drawExtrudedContour(contour,
        Point3D<float>(-60,60,0), //Position
        Point3D<float>(rot,rot,0), //Rotation
        30.0F,
        PixRGB<byte>(256,0,0));


    //Test the box display
    vp.drawBox( Point3D<float>(60,60,0), //Position
        Point3D<float>(rot,rot,0), //Rotation
        Point3D<float>(30, 30, 50), //size
        PixRGB<byte>(0,256,0)
        );

    //Test the sphere display
    vp.drawSphere( Point3D<float>(60,-60,0), //Position
        Point3D<float>(rot,rot,0), //Rotation
        Point3D<float>(30, 30, 60), //size
        PixRGB<byte>(0,0,256)
        );

    //Test the cylinder display
    vp.drawCone( Point3D<float>(-60,-60,0), //Position
        Point3D<float>(rot,rot,0), //Rotation
        30, //radius
        50, //length
        PixRGB<byte>(256,256,0)
        );

    Image<PixRGB<byte> > img = flipVertic(vp.getFrame());
    ofs->writeRGB(img, "ViewPort3D", FrameInfo("ViewPort3D", SRC_POS));
    usleep(10000);
  }


  exit(0);

}


