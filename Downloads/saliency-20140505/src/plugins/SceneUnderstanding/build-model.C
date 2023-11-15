/*!@file SceneUnderstanding/build-model.C Build a 3D model */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/build-model.C $
// $Id: build-model.C 13051 2010-03-25 01:29:33Z lior $


#include "GUI/ViewPort3D.H"
#include "Util/log.H"
#include "Util/WorkThreadServer.H"
#include "Util/JobWithSemaphore.H"
#include "Component/ModelManager.H"
#include "Raster/GenericFrame.H"
#include "Image/Layout.H"
#include "Image/MatrixOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/MathOps.H"
#include "Image/PixelsTypes.H"
#include "Image/ColorOps.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "GUI/XWinManaged.H"
#include "GUI/ImageDisplayStream.H"

#include <stdlib.h>
#include <math.h>
#include <fcntl.h>

void drawAir (void) {
  //int stencil;

  ///* Find out current stencil function. */
  //glGetIntegerv (GL_STENCIL_FUNC, &stencil);
  //
  ///* Describe airplane using simple concave polygons.
  // * See `airplane-blueprint' for details. */
  //glStencilFunc (stencil, 1, 0xf);
  glBegin (GL_POLYGON); /*nose*/
  glEdgeFlag (1);
  glVertex3f (0,-1, 10); 
  glEdgeFlag (0);
  glVertex3f (2, 0,  4);
  glEdgeFlag (1);
  glVertex3f (-2, 0,  4);  
  glEnd ();

  glEdgeFlag (1);
  ////glStencilFunc (stencil, 2, 0xf);
  //glBegin (GL_POLYGON); /*tail*/
  //glEdgeFlag (1);
  //glVertex3f ( 1, 0, -3);
  //glVertex3f ( 3, 0, -4);
  //glVertex3f ( 3, 0, -5);  
  //glVertex3f (-3, 0, -5);
  //glVertex3f (-3, 0, -4);
  //glEdgeFlag (0);
  //glVertex3f (-1, 0, -3);
  //glEnd ();

  //glBegin (GL_POLYGON); /*body 1/2*/
  //glEdgeFlag (0);
  //glVertex3f ( 2, 0,  4);
  //glEdgeFlag (1);
  //glVertex3f ( 2, 0, -.5);
  //glVertex3f ( 2, 0, -1);
  //glEdgeFlag (0);
  //glVertex3f ( 1, 0, -1);
  //glEdgeFlag (1);
  //glVertex3f (-1, 0, -1);
  //glVertex3f (-2, 0, -1);
  //glEdgeFlag (0);
  //glVertex3f (-2, 0, -.5);
  //glVertex3f (-2, 0,  4);
  //glEnd ();
  
  //glBegin (GL_POLYGON); /*body 2/2*/
  //glEdgeFlag (1);
  //glVertex3f ( 1, 0, -1);
  //glEdgeFlag (0);
  //glVertex3f ( 1, 0, -3);
  //glEdgeFlag (1);
  //glVertex3f (-1, 0, -3);
  //glEdgeFlag (0);
  //glVertex3f (-1, 0, -1);
  //glEnd ();
  //
  //glBegin (GL_POLYGON); /*left wingtip*/
  //glEdgeFlag (1);
  //glVertex3f ( 8, 0,  1);
  //glVertex3f ( 8, 0, -1);
  //glVertex3f ( 5, 0, -1);
  //glEdgeFlag (0);
  //glVertex3f ( 5, 0, -.5);
  //glEnd ();
  //
  //glBegin (GL_POLYGON); /*left wing*/
  //glEdgeFlag (1);
  //glVertex3f ( 2, 0,  4);
  //glEdgeFlag (0);
  //glVertex3f ( 8, 0,  1);
  //glEdgeFlag (1);
  //glVertex3f ( 5, 0, -.5);
  //glEdgeFlag (0);
  //glVertex3f ( 2, 0, -.5);
  //glEnd ();
  //
  //glBegin (GL_POLYGON); /*right wingtip*/
  //glEdgeFlag (1);
  //glVertex3f (-5, 0, -.5);
  //glVertex3f (-5, 0, -1);
  //glVertex3f (-8, 0, -1);
  //glEdgeFlag (0);
  //glVertex3f (-8, 0,  1);
  //glEnd ();

  //glBegin (GL_POLYGON); /*right wing*/
  //glEdgeFlag (0);
  //glVertex3f (-2, 0,  4);
  //glEdgeFlag (1);
  //glVertex3f (-2, 0, -.5);
  //glEdgeFlag (0);
  //glVertex3f (-5, 0, -.5);
  //glEdgeFlag (1);
  //glVertex3f (-8, 0,  1);
  //glEnd ();
  //
  ///* Create rotated coordinate system for left flap. */
  //glPushMatrix ();
  //glTranslatef (2, 0, -.5);
  //glRotatef (0, 1, 0, 0);
  //
  ////glStencilFunc (stencil, 3, 0xf);
  //glBegin (GL_POLYGON); /*left flap*/
  //glEdgeFlag (1);
  //glVertex3f ( 3, 0,  0);
  //glVertex3f ( 0, 0,  0);
  //glVertex3f ( 0, 0, -1);
  //glVertex3f ( 3, 0, -1);
  //glEnd ();
  //glPopMatrix ();
  //
  ///* Create rotated coordinate system for right flap. */
  //glPushMatrix ();
  //glTranslatef ( -2, 0, -.5);
  //glRotatef (0, 1, 0, 0);
  //
  ////glStencilFunc (stencil, 4, 0xf);
  //glBegin (GL_POLYGON); /* right flap */
  //glEdgeFlag (1);
  //glVertex3f (-3, 0,  0);
  //glVertex3f ( 0, 0,  0);
  //glVertex3f ( 0, 0, -1);
  //glVertex3f (-3, 0, -1);
  //glEnd ();
  //glPopMatrix ();

  ///* Create coordinate system for tail wing. */
  //glPushMatrix ();
  //glTranslatef (0, 0, -4.5);

  ////glStencilFunc (stencil, 5, 0xf); /*tail wing*/
  //glBegin (GL_POLYGON);
  //glEdgeFlag (0);
  //glVertex3f (0, 0,   0);
  //glEdgeFlag (1);
  //glVertex3f (0, 1,   1.5);
  //glVertex3f (0, 0,   3);
  //glEnd ();
  //glBegin (GL_POLYGON);
  //glEdgeFlag (1);
  //glVertex3f (0, 0,    0);
  //glEdgeFlag (0);
  //glVertex3f (0, 2.5,  0);
  //glEdgeFlag (1);
  //glVertex3f (0, 3,   .5);
  //glEdgeFlag (0);
  //glVertex3f (0, 1,   1.5);
  //glEnd ();
  //glBegin (GL_POLYGON);
  //glEdgeFlag (1);
  //glVertex3f (0, 2.5,  0);
  //glVertex3f (0, 2.5, -.5);
  //glVertex3f (0, 3,   -.5);
  //glEdgeFlag (0);
  //glVertex3f (0, 3,   .5);
  //glEnd ();

  ///* Create coordinate system for rudder. */
  //glRotatef (0, 0, 1, 0);
  //
  ////glStencilFunc (stencil, 6, 0xf);
  //glBegin (GL_POLYGON); /*rudder*/
  //glEdgeFlag (1);
  //glVertex3f (0, 0,    0);
  //glVertex3f (0, 2.5,  0);
  //glVertex3f (0, 2.5, -1);
  //glVertex3f (0, 0,   -1);
  //glEnd ();
  //glPopMatrix ();
  //
  ////glStencilFunc (stencil, 7, 0xf);
  //glBegin (GL_POLYGON); /*cockpit right front*/
  //glEdgeFlag (1);
  //glVertex3f ( 0, -1, 10);
  //glVertex3f (-2,  0,  4);
  //glVertex3f ( 0,  1.5,5);
  //glEnd ();
  ////glStencilFunc (stencil, 8, 0xf);
  //glBegin (GL_POLYGON); /*cockpit left front*/
  //glEdgeFlag (1);
  //glVertex3f ( 0, -1, 10);
  //glVertex3f ( 0,  1.5, 5);
  //glVertex3f ( 2,  0,   4);
  //glEnd ();
  ////glStencilFunc (stencil, 9, 0xf);
  //glBegin (GL_POLYGON); /*cockpit left back*/
  //glEdgeFlag (1);
  //glVertex3f ( 0,  1.5, 5);
  //glVertex3f ( 2,  0,   4);
  //glVertex3f ( 1,  0,  -1);
  //glEnd ();
  ////glStencilFunc (stencil, 10, 0xf);
  //glBegin (GL_POLYGON); /*cockpit right back*/
  //glEdgeFlag (1);
  //glVertex3f (-2,  0,   4);
  //glVertex3f ( 0,  1.5, 5);
  //glVertex3f (-1,  0,  -1);
  //glEnd ();
  ////glStencilFunc (stencil, 11, 0xf);
  //glBegin (GL_POLYGON); /*cocpit top*/
  //glEdgeFlag (1);
  //glVertex3f ( 0,  1.5, 5);
  //glEdgeFlag (0);
  //glVertex3f (-1,  0,  -1);
  //glEdgeFlag (1);
  //glVertex3f ( 1,  0,  -1);
  //glEnd ();
}

void tridisplay()
// this function is used to display the polygon using triangles
// it is not efficient or neat but it is necessary because the
// assignment asks for a polygon drawn with triangles and without
// diagonals
//
// this function produces the same result as redisplay() with GL_LINE_LOOP

{
  glClear(GL_COLOR_BUFFER_BIT); 

  glBegin(GL_TRIANGLES);
  
  glColor3f(0.3,0.3,0.3);//navy
  glVertex2f(-0.5,0.0);
  glColor3f(0.0,0.0,1.0);//blue
  glVertex2f(0.0,-1.0);
  glEdgeFlag(GL_FALSE);
  glVertex2f(0.5,0.0);
  
  glEdgeFlag(GL_TRUE);
  glVertex2f(0.5,0.0);
  glColor3f(0.0,1.0,0.0);//green
  glVertex2f(0.497,0.052);
  glEdgeFlag(GL_FALSE);  
  glVertex2f(0.489,0.104);

  glEdgeFlag(GL_TRUE);
  glVertex2f(0.489,0.104);
  glVertex2f(0.476,0.155);
  glEdgeFlag(GL_FALSE);
  glVertex2f(0.457,0.203);

  glEdgeFlag(GL_TRUE);
  glVertex2f(0.457,0.203);
  glVertex2f(0.433,0.25);
  glEdgeFlag(GL_FALSE);
  glVertex2f(0.405,0.294);

  glEdgeFlag(GL_TRUE);
  glVertex2f(0.405,0.294);
  glVertex2f(0.372,0.335);
  glColor3f(1.0,0.0,0.0);//red
  glEdgeFlag(GL_FALSE);
  glVertex2f(0.335,0.372);

  glEdgeFlag(GL_TRUE);
  glVertex2f(0.335,0.372);
  glVertex2f(0.294,0.405);
  glEdgeFlag(GL_FALSE);
  glVertex2f(0.25,0.433);

  glEdgeFlag(GL_TRUE);
  glVertex2f(0.25,0.433);
  glVertex2f(0.203,0.457);
  glEdgeFlag(GL_FALSE);
  glVertex2f(0.155,0.476);

  glEdgeFlag(GL_TRUE);
  glVertex2f(0.155,0.476);
  glVertex2f(0.104,0.489);
  glEdgeFlag(GL_FALSE);
  glVertex2f(0.052,0.497);

  glEdgeFlag(GL_TRUE);
  glVertex2f(0.052,0.497);
  glVertex2f(0.0,0.5);
  glColor3f(0.5,0.0,0.5);//purple
  glEdgeFlag(GL_FALSE);
  glVertex2f(-0.052,0.497);

  glEdgeFlag(GL_TRUE);
  glVertex2f(-0.052,0.497);
  glVertex2f(-0.104,0.489);
  glEdgeFlag(GL_FALSE);
  glVertex2f(-0.155,0.476);

  glEdgeFlag(GL_TRUE);
  glVertex2f(-0.155,0.476);
  glVertex2f(-0.203,0.457);
  glEdgeFlag(GL_FALSE);
  glVertex2f(-0.25,0.433);

  glEdgeFlag(GL_TRUE);
  glVertex2f(-0.25,0.433);
  glVertex2f(-0.294,0.405);
  glEdgeFlag(GL_FALSE);
  glVertex2f(-0.335,0.372);

  glEdgeFlag(GL_TRUE);
  glVertex2f(-0.335,0.372);
  glVertex2f(-0.372,0.335);
  glEdgeFlag(GL_FALSE);
  glVertex2f(-0.404,0.294);

  glEdgeFlag(GL_TRUE);
  glVertex2f(-0.404,0.294);
  glVertex2f(-0.433,0.25);
  glEdgeFlag(GL_FALSE);
  glVertex2f(-0.457,0.203);

  glEdgeFlag(GL_TRUE);
  glVertex2f(-0.457,0.203);
  glVertex2f(-0.476,0.155);
  glEdgeFlag(GL_FALSE);
  glVertex2f(-0.489,0.104);

  glEdgeFlag(GL_TRUE);
  glVertex2f(-0.489,0.104);
  glVertex2f(-0.497,0.052);
  glEdgeFlag(GL_FALSE);
  glVertex2f(-0.5,0.0);
  glEdgeFlag(GL_TRUE);
  
  glEnd();
  
  glFlush();
}


Point2D<int> getMouseClick(nub::ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("InputEdges")
    : rutz::shared_ptr<XWinManaged>();

  if (uiwin.is_valid())
    return uiwin->getLastMouseClick();
  else
    return Point2D<int>(-1,-1);
}

int getKey(nub::ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("InputEdges")
    : rutz::shared_ptr<XWinManaged>();
  if (uiwin.is_valid())
    return uiwin->getLastKeyPress();
  else
    return -1;
}

void saveToDisk(const std::vector<Point2D<float> >& contour, const char* filename)
{
  int fd;

  if ((fd = creat(filename, 0644)) == -1)
    LFATAL("Can not open %s for saving\n", filename);

  int ret;
  size_t size = contour.size();
  ret = write(fd, (char *) &size, sizeof(size_t));

  for(uint i=0; i<contour.size(); i++)
    ret = write(fd, (char *) &contour[i], sizeof(Point2D<float>));

  close(fd);
}

std::vector<Point2D<float> > readFromDisk(const char* filename)
{

  int fd;
  if ((fd = open(filename, 0, 0644)) == -1)
    return std::vector<Point2D<float> >();

  int ret;
  printf("Reading from %s\n", filename);

  //Read the number of entries in the db
  size_t size;
  ret = read(fd, (char *) &size, sizeof(size));

  std::vector<Point2D<float> > contour(size);
  for(uint i=0; i<size; i++)
    ret = read(fd, (char *) &contour[i], sizeof(Point2D<float>));

  close(fd);

  return contour;
}





int main(int argc, char *argv[]){

  ModelManager manager("Test Viewport");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);


  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);
  // let's get all our ModelComponent instances started:
  manager.start();

  ifs->startStream();

  ViewPort3D vp(320,240); //, true, false, true);
  //vp.setWireframeMode(true);
  //vp.setLightsMode(false);
  //vp.setFeedbackMode(true);

  std::vector<Point2D<int> > polygon;
  std::vector<Point2D<float> > contour;
  std::vector<Point2D<float> > bodyContour = readFromDisk("body.dat");
  std::vector<Point2D<float> > roll1Contour = readFromDisk("roll1.dat");
  std::vector<Point2D<float> > roll2Contour = readFromDisk("roll2.dat");

  //vp.setCamera(Point3D<float>(0,150,354), Point3D<float>(20,00,0));
  //double trans[3][4] = {
  //  {0.999113, 0.007695, 0.041394, 6.445829},
  //  {0.006331, -0.999436, 0.032987, -4.874230},
  //  {0.041625, -0.032696, -0.998598, 182.532963}};
  //vp.setCamera(trans);
  float x = -130;
  while(1)
  {
    vp.setCamera(Point3D<float>(0,0,0), Point3D<float>(0,0,0));
    Image< PixRGB<byte> > inputImg;
    const FrameState is = ifs->updateNext();
    if (is == FRAME_COMPLETE)
      break;
    GenericFrame input = ifs->readFrame();
    if (!input.initialized())
      break;
    inputImg = input.asRgb();

    int key = getKey(ofs);
    if (key != -1)
    {
      LINFO("key %i", key);
      switch(key)
      {
        case 98: //up
          x += 1;
          break;
        case 104: //down
          x -= 1;
          break;
        case 39: //s
          saveToDisk(contour, "body.dat");
          break;
      }
      LINFO("X=%f\n", x);
    }

    vp.initFrame();

    //vp.drawGround(Point2D<float>(1000,1000),
    //    PixRGB<byte>(255,255,255));

    std::vector<Point2D<float> > triangles = vp.triangulate(contour);

    //vp.drawBox(
    //    Point3D<float>(0,0,x),
    //    Point3D<float>(0,0,0),
    //    Point3D<float>(41.5, 41.5, 0.001),
    //    PixRGB<byte>(255,255,255));

    
    //glTranslatef(0.0F, 0.0F,-130);
    //glBegin(GL_TRIANGLES);
    //for(uint i=0; i<triangles.size(); i++)
    //{
    //  glVertex3f(triangles[i].i, triangles[i].j,0.0f);                        // Top Right Of The Quad (Top)
    //}
    //glEnd();

    glTranslatef(0.0F, 0,-130-25.0F);
    glRotatef(x, 0,1,0);
    if (true)
    {
      vp.drawExtrudedContour(bodyContour,
          Point3D<float>(0,0,0),
          Point3D<float>(0,0,0),
          50.0F,
          PixRGB<byte>(256,256,0));

    }

    Point2D<int> loc = getMouseClick(ofs);
    if (loc.isValid())
    {
      polygon.push_back(loc);

      float x = loc.i - (inputImg.getWidth()/2);
      float y = -1*(loc.j - (inputImg.getHeight()/2));
      LINFO("Click at %i %i (%f,%f)\n", loc.i, loc.j, x, y);
      contour.push_back(Point2D<float>(x/2.72289,y/2.72289));
    }


    Image<PixRGB<byte> > img = flipVertic(vp.getFrame()) + inputImg;
    //Image<PixRGB<byte> > img = inputImg;

    //std::vector<ViewPort3D::Line> lines = vp.getFrameLines();
    //for(uint i=0; i<lines.size(); i++)
    //{
    //  LINFO("Line %f %f  %f %f", 
    //      lines[i].p1.i, lines[i].p1.j, 
    //      lines[i].p2.i, lines[i].p2.j);
    //}

    Image<PixRGB<byte> > renderImg = flipVertic(vp.getFrame());
    //img = renderImg;
    for(uint i=0; i<img.size(); i++)
      if (renderImg[i] != PixRGB<byte>(0,0,0))
        img[i] = renderImg[i];

    ofs->writeRGB(img, "ViewPort3D", FrameInfo("ViewPort3D", SRC_POS));
    ofs->writeRGB(inputImg, "Input", FrameInfo("Input", SRC_POS));

    Image<float> mag, ori, lum;
    lum = luminance(renderImg);
    gradientSobel(lum, mag, ori);
    inplaceNormalize(mag, 0.0F, 255.0F);
    ofs->writeRGB(toRGB(mag), "RenderEdges", FrameInfo("RenderEdges", SRC_POS));

    lum = luminance(inputImg);
    gradientSobel(lum, mag, ori);
    inplaceNormalize(mag, 0.0F, 255.0F);
    ofs->writeRGB(toRGB(mag), "InputEdges", FrameInfo("InputEdges", SRC_POS));


    usleep(10000);
  }


  exit(0);

}



