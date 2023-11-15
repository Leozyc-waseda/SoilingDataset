/*!@file Qt/beobotmap.cpp */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/beobotmap.cpp $
// $Id: beobotmap.cpp 5735 2005-10-18 16:00:27Z rjpeters $
//

#include "beobotmap.h"
#include <cmath>

BeoBotMap::BeoBotMap(QWidget *parent, const char *name)
                :QGLWidget(parent, name)
{
        setFormat(QGLFormat(DoubleBuffer | DepthBuffer));
        // init position
        iconlistx=100; iconlisty=100;
        iconlistgx = 0; iconlistgy= 0;
        colors[0] = yellow;
        colors[1] = blue;
        colors[2] = green;
        colors[3] = white;
        colors[4] = red;
        colors[5] = QColor(255,85,0);
        colors[6] = QColor(0,170,255);

        // create the icon
        createIcon(2);
}

void BeoBotMap::createIcon(int num)
{
  int i;
  sizeOfList = num;

  listOfIcon = (BeoBotIcon *)malloc(sizeof(BeoBotIcon)*sizeOfList);
  for(i=0 ; i<sizeOfList ; i++)
  {
    listOfIcon[i].x = 100;
    listOfIcon[i].y = 100;
    listOfIcon[i].glx = 0.0;
    listOfIcon[i].gly = 0.0;
    listOfIcon[i].edgeSize = 30;
    listOfIcon[i].selected = 0;

    // FIXME for test. should be delete after finishing
    if(i==1)
    {
    listOfIcon[i].x = 140;
    listOfIcon[i].y = 140;
    listOfIcon[i].glx = 1.0;
    listOfIcon[i].gly = -1.0;
    listOfIcon[i].edgeSize = 30;
    listOfIcon[i].selected = 0;

    }
  }
  updateGL();
}

void BeoBotMap::initializeGL()
{
        qglClearColor(black);
        glShadeModel(GL_FLAT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
}

void BeoBotMap::resizeGL(int w, int h)
{
        glViewport(0, 0, w, h);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        GLfloat x = (GLfloat)w/h;
        glFrustum(-x, x, -1.0, 1.0, 4.0, 15.0);
        glMatrixMode(GL_MODELVIEW);
}

void BeoBotMap::paintGL()
{
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        draw();
}

void BeoBotMap::draw()
{
    int i,j;

              GLfloat coords[4][2] =
          {{-0.125,-0.125},{0.125,-0.125},
           {0.125,0.125},{-0.125,0.125}};  // define the view point

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0,0.0,-10.0);

        // drawgl the objects
        qglColor(red);
        glBegin(GL_QUADS);
        for(i=0 ; i<sizeOfList ; i++)
        for(j=0 ; j<4 ; j++)
          glVertex3f(coords[j][0]+listOfIcon[i].glx,coords[j][1]+listOfIcon[i].gly,0.0);
        glEnd();

        // FIXME just for testing
        qglColor(white);
        glBegin(GL_TRIANGLES);
        glVertex2f(0.0,0.1);
        glVertex2f(-0.1,-0.1);
        glVertex2f(0.1,-0.1);
        glEnd();
        glBegin(GL_QUADS);
        glVertex2f(-0.025,-0.2);
        glVertex2f(0.025,-0.2);
        glVertex2f(0.025,-0.1);
        glVertex2f(-0.025,-0.1);
        glEnd();

}

void BeoBotMap::mousePressEvent(QMouseEvent *event)
{
        lastPos = event->pos();
   //     printf("press:%d,%d\n", event->x(),event->y());
}

void BeoBotMap::mouseDoubleClickEvent(QMouseEvent *event)
{
        int i;
        lastPos = event->pos();
        // double click to select the desired square

        // find the selected one in the list
        for(i=0 ; i<sizeOfList; i++)
        {
          if(lastPos.x() < (listOfIcon[i].x)+15 &&
             lastPos.x() > (listOfIcon[i].x)-15 &&
             lastPos.y() < (listOfIcon[i].y)+15 &&
             lastPos.y() > (listOfIcon[i].y)-15
            )
                listOfIcon[i].selected = 1;
          else
                listOfIcon[i].selected = 0;
        }
}

void BeoBotMap::mouseReleaseEvent(QMouseEvent *event)
{
}

void BeoBotMap::mouseMoveEvent(QMouseEvent *event)
{
        int dx = event->x() - lastPos.x();
        int dy = event->y() - lastPos.y();
        lastPos = event->pos();
 //       printf("press on and move:%d,%d\n", event->x(),event->y());
          if(event->state() & LeftButton)
          {
            for(int i=0 ; i<sizeOfList ; i++)
            if(lastPos.x() < (listOfIcon[i].x)+15 &&
               lastPos.x() > (listOfIcon[i].x)-15 &&
               lastPos.y() < (listOfIcon[i].y)+15 &&
               lastPos.y() > (listOfIcon[i].y)-15 &&
               listOfIcon[i].selected == 1
              )
            {
              listOfIcon[i].x += dx;
              listOfIcon[i].y += dy;
              listOfIcon[i].glx += GLfloat(dx) * 0.025;
              listOfIcon[i].gly -= GLfloat(dy) * 0.025;
               // updateGL();

       //         printf("in and move!");
            }
          }
          updateGL();
}

void BeoBotMap::returnSelectedCoord(float &fx, float &fy)
{
        int i;
        fx = fy = 0.0F;
        for(i=0 ; i<sizeOfList ; i++)
          if(listOfIcon[i].selected == 1)
          {
             fx = listOfIcon[i].x;
             fy = listOfIcon[i].y;
          }
}
