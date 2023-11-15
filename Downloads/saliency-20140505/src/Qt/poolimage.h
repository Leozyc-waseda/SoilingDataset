/*!@file Qt/poolimage.h */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/poolimage.h $
// $Id: poolimage.h 12269 2009-12-17 01:23:34Z itti $
//

#ifndef QT_POOL_IMAGE_H_DEFINED
#define QT_POOL_IMAGE_H_DEFINED

#ifdef INVT_HAVE_QT3

#include <qgl.h>

class iconImage
{
        public:
                iconImage(){};
                iconImage(int ax, int ay,int size)
                {
                        x = ax; y = ay;
                        edgeSize = size;
                        selected = false;
                        glx = (x - 300) * 0.01137; gly = (y - 220) * 0.01137;
                };
                void setPosition(int ax, int ay)
                {
                        x = x + ax; y = y + ay;
                        glx = glx + GLfloat(ax) * 0.01137;
                        gly = gly - GLfloat(ay) * 0.01137;
                };
                iconImage operator=(const iconImage other)
                {
                        iconImage temp;
                        temp.x = other.x;
                        temp.y = other.y;
                        temp.glx = other.glx;
                        temp.gly = other.gly;
                        temp.selected = other.selected;
                        temp.edgeSize = other.edgeSize;
                        return temp;
                };
                int x, y;
                GLfloat glx, gly;

                int selected;
                int edgeSize;
};
class PoolImage : public QGLWidget
{
        public:
                PoolImage(QWidget * parent = 0, const char *name = 0);
                void createIcon(int num);
                void currentItem(int item);
                float getAngle();
                void getCoord(int i, int &x, int &y);
                //void setAngle(float a);
                void setCoordByPix(int i, int x, int y);
                void setCoordByGL(int i, float glx, float gly);
                //void setScale(float a);
                void getRealPosition(int i, float &rx, float &ry);
                void getGLPosition(float rx, float ry, float &gx, float &gy);

                inline void reset(){free(iconlist); sizeOfList = 0;};
                inline iconImage *getList(){return iconlist;};

                float angle;
                float scale;
        protected:
                void initializeGL();
                void resizeGL(int w, int h);
                void paintGL();

                void mousePressEvent(QMouseEvent *event);
                void mouseMoveEvent(QMouseEvent *event);
                void mouseDoubleClickEvent(QMouseEvent *event);
                void mouseReleaseEvent(QMouseEvent *event);
        private:
                void draw();
                bool right;
                QColor colors[7];
                QPoint lastPos;
                QPoint dragPos;
                int sizeOfList;
                int maxX, minX, maxY, minY;
                iconImage *iconlist;
};

#endif // INVT_HAVE_QT3

#endif // QT_POOL_IMAGE_H_DEFINED

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

