/*!@file Qt/beobotmap.h */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/beobotmap.h $
// $Id: beobotmap.h 5735 2005-10-18 16:00:27Z rjpeters $
//

#include <qgl.h>
class BeoBotIcon
{
        public:
                BeoBotIcon(){};
                BeoBotIcon(int ax, int ay,int size)
                {
                        x = ax; y = ay;
                        edgeSize = size;
                        selected = false;
                        glx = (x - 100) * 0.025; gly = (y - 100) * 0.025;
                };
                void setPosition(int ax, int ay)
                {
                        x = x + ax; y = y + ay;
                        glx = glx + GLfloat(ax) * 0.025;
                        gly = gly - GLfloat(ay) * 0.025;
                };
                BeoBotIcon operator=(const BeoBotIcon other)
                {
                        BeoBotIcon temp;
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
class BeoBotMap : public QGLWidget
{
        public:
                BeoBotMap(QWidget * parent = 0, const char *name = 0);

                void returnSelectedCoord(float &fx, float &fy);
        protected:
                void initializeGL();
                void resizeGL(int w, int h);
                void paintGL();

                void createIcon(int num);
                void mousePressEvent(QMouseEvent *event);
                void mouseMoveEvent(QMouseEvent *event);
                void mouseDoubleClickEvent(QMouseEvent *event);
                void mouseReleaseEvent(QMouseEvent *event);
        private:
                int sizeOfList;
                void draw();
                float iconlistx, iconlisty;
                float iconlistgx, iconlistgy;
                BeoBotIcon *listOfIcon;
                QPoint lastPos;
                QPoint dragPos;
                QColor colors[7];
};
