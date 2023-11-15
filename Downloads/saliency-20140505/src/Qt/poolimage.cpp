/*!@file Qt/poolimage.cpp */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/poolimage.cpp $
// $Id: poolimage.cpp 12269 2009-12-17 01:23:34Z itti $
//


#ifndef QT_POOL_IMAGE_C_DEFINED
#define QT_POOL_IMAGE_C_DEFINED

#ifdef INVT_HAVE_QT3

#include "poolimage.h"
#include <cmath>

PoolImage::PoolImage(QWidget *parent, const char *name)
                :QGLWidget(parent, name)
{
        setFormat(QGLFormat(DoubleBuffer | DepthBuffer));
        sizeOfList = 0;
        angle = 0.0F;
        scale = 1.0F;
        right = false;
        maxX=minX=maxY=minY=0;

        colors[0] = yellow;
        colors[1] = blue;
        colors[2] = green;

        colors[3] = white;
        colors[4] = red;
        colors[5] = QColor(255,85,0);
        colors[6] = QColor(0,170,255);
}


void PoolImage::initializeGL()
{
        qglClearColor(black);
        glShadeModel(GL_FLAT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
}

void PoolImage::resizeGL(int w, int h)
{
        glViewport(0, 0, w, h);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        GLfloat x = (GLfloat)w/h;
        glFrustum(-x, x, -1.0, 1.0, 4.0, 15.0);
        glMatrixMode(GL_MODELVIEW);
}

void PoolImage::paintGL()
{
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        draw();
}

void PoolImage::draw()
{
        int theone=-1, i, ci;

        static const GLfloat coords[4][2] =
        {{-0.125,-0.125},{0.125,-0.125},
         {0.125,0.125},{-0.125,0.125}};

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0,0.0,-10.0);
        // draw the selected icon
          for(ci=0; ci<sizeOfList ; ci++)
          {
          if(iconlist[ci].selected == 2)
          {
            glBegin(GL_QUADS);
            qglColor(colors[iconlist[ci].selected]);
            for(i=0 ; i<4 ; i++)
                glVertex3f(coords[i][0]+(iconlist[ci].glx), coords[i][1]+(iconlist[ci].gly), 0.0);
            glEnd();
            theone = ci;
          }
          }

        for(ci=0; ci<sizeOfList ; ci++)
        {
          if(ci!=theone)
          {
            glBegin(GL_QUADS);
            qglColor(colors[iconlist[ci].selected]);
            for(i=0 ; i<4 ; i++)
              glVertex3f(coords[i][0]+(iconlist[ci].glx), coords[i][1]+(iconlist[ci].gly), 0.0);
            glEnd();
          }
        }

        // draw the line with certain rotation which can be set from the gui
        qglColor(white);
        glRotatef(0+angle,0,0,1);
        for(i=0 ; i<40 ; i++)
        {
          glBegin(GL_LINES);
          glVertex2f(-5.0+i*0.25, -5.0);
          glVertex2f(-5.0+i*0.25,5.0);
          glEnd();
        }

        //glTranslate(1.0,1.0,0.0);
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

        glRotatef(90,0,0,1);
        for(i=0 ; i<40 ; i++)
        {
          glBegin(GL_LINES);
          glVertex2f(-5.0+i*0.25, -5.0);
          glVertex2f(-5.0+i*0.25,5.0);
          glEnd();
        }

}

void PoolImage::mousePressEvent(QMouseEvent *event)
{
        lastPos = event->pos();
}

void PoolImage::mouseDoubleClickEvent(QMouseEvent *event)
{
/*
        dragPos = event->pos();
        minX = event->x();
        minY = event->y();
        printf("dclick:%d,%d\n", minX, minY);
*/
}

void PoolImage::mouseReleaseEvent(QMouseEvent *event)
{
/*
        maxX = event->x();
        maxY = event->y();
        printf("mis::%d,%d\n", dragPos.x(), dragPos.y());
        printf("rclick:%d,%d\n", maxX, maxY);
        for(int i=0 ; i<sizeOfList ; i++)
          if(iconlist[i].selected == 2)
                  iconlist[i].selected = 1;
        for(int i=0 ; i<sizeOfList ; i++)
        {
          if(iconlist[i].x> dragPos.x()&& iconlist[i].x<maxX &&
             iconlist[i].y> dragPos.y()&& iconlist[i].y<maxY)
          {
                  iconlist[i].selected = 2;
          }
        }
        //minX = maxX = minY = maxY = 0;
        updateGL();
*/
}

void PoolImage::mouseMoveEvent(QMouseEvent *event)
{
        int dx = event->x() - lastPos.x();
        int dy = event->y() - lastPos.y();
        lastPos = event->pos();
        // check every element in the list
        for(int i=0 ; i<sizeOfList ; i++)
        {
          if(event->state() & LeftButton)
          {
            if(//lastPos.x() < (iconlist[i].x)+(iconlist[i].edgeSize)/2 &&
               //lastPos.x() > (iconlist[i].x)-(iconlist[i].edgeSize)/2 &&
               //lastPos.y() < (iconlist[i].y)+(iconlist[i].edgeSize)/2 &&
               //lastPos.y() > (iconlist[i].y)-(iconlist[i].edgeSize)/2 &&
               iconlist[i].selected ==2)
            {
                iconlist[i].setPosition(dx, dy);
                updateGL();
            }
          }
          else if(event->state() & RightButton)
          {
          }

        }
}

void PoolImage::createIcon(int num)
{
        int i;
        int oldSize = sizeOfList;
        sizeOfList = num;

        // if there are previous generated iconImage objects
        if(oldSize != 0 && oldSize <= sizeOfList)
        {
          iconImage *temp = (iconImage *)malloc(sizeof(iconImage)*(sizeOfList+4));
          for(i=0 ; i<oldSize ; i++)
                temp[i] = iconlist[i];
          for(i = oldSize ; i<sizeOfList ; i++)
          {
                iconlist[i].x = 300;
                iconlist[i].y = 220;
                iconlist[i].glx = 0.0;
                iconlist[i].gly = 0.0;
                iconlist[i].edgeSize = 22;
                iconlist[i].selected = 0;
          }
        }
        else
        {
          iconlist = (iconImage *)malloc(sizeof(iconImage)*sizeOfList);
            for(i=0; i<sizeOfList ; i++)
          {
                // for the tasks and the gate
                  iconlist[i].x = 300;
                  iconlist[i].y = 220;
                  iconlist[i].glx = 0.0;
                iconlist[i].gly = 0.0;
                  iconlist[i].edgeSize = 22;
                 iconlist[i].selected = 0;
          }
        }
          updateGL();
}

void PoolImage::currentItem(int item)
{
        for(int i=0 ; i<sizeOfList ; i++)
        {
          if(i == item)
                  iconlist[i].selected = 2;
          else
          {
                if( i < 4)
                  iconlist[i].selected = 3+i;
                else
                  iconlist[i].selected = 1;
          }
        }
        updateGL();
}

void PoolImage::getCoord(int i, int &x, int &y)
{
        x = iconlist[i].x;
        y = iconlist[i].y;
}

void PoolImage::setCoordByPix(int i, int x, int y)
{
        iconlist[i].x = x;
        iconlist[i].y = y;
        iconlist[i].glx = (GLfloat)(x-300) * 0.01137F;
        iconlist[i].gly = -(GLfloat)(y-220) * 0.01137F;
        updateGL();
}

void PoolImage::setCoordByGL(int i, float glx, float gly)
{
        iconlist[i].glx = glx;
        iconlist[i].gly = gly;
        iconlist[i].x = int(glx/0.01137F + 300.0F + 0.5F);
        iconlist[i].y = int(gly/0.01137F + 220.0F + 0.5F);
        updateGL();
}

float PoolImage::getAngle()
{
  return angle;
}

void PoolImage::getRealPosition(int i, float &rx, float &ry)
{
  // translate to the original point
  float tempx = float(iconlist[i].glx - iconlist[0].glx);
  float tempy = -float(iconlist[0].gly - iconlist[i].gly);
  float tempa = angle;

  if(i == 0)
  {
          tempx = iconlist[0].glx;
          tempy = iconlist[0].gly;
  }
  // nomalize
  while(tempa > 180)
          tempa = tempa -360;
  while(tempa < -180)
          tempa = tempa + 360;
  tempa = -tempa/180.0F * 3.14F;

  // rotate and scale
  rx = (tempx * cos(tempa) - tempy * sin(tempa)) * scale;
  ry = (tempx * sin(tempa) + tempy * cos(tempa)) * scale;
  /*
  if( i == 0)
  {
          rx = 0.0F; ry = 0.0F;
  }
  */
}

void PoolImage::getGLPosition(float rx, float ry, float &gx, float &gy)
{
  // scale
  float tx, tempx = rx / scale;
  float ty, tempy = ry / scale;
  float tempa = angle;

  // nomalize
  while(tempa > 180)
          tempa = tempa -360;
  while(tempa < -180)
          tempa = tempa + 360;
  tempa = tempa/180.0F * 3.14F;

  // rotate
  tx = tempx * cos(tempa) - tempy * sin(tempa);
  ty = tempx * sin(tempa) + tempy * cos(tempa);

  // translate
  gx = tx + iconlist[0].glx;
  gy = ty + iconlist[0].gly;

}

#endif // INVT_HAVE_QT3

#endif // QT_POOL_IMAGE_C_DEFINED

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
