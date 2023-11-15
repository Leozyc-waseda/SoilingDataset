/*!@file Qt4/ImageView.qt.H */

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
// Primary maintainer for this file: Pezhman Firoozfam (pezhman.firoozfam@usc.edu)
// $HeadURL$ svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt4/ImageView.qt.H $
//

#ifndef QT4_IMAGEVIEW_QT_H_DEFINED
#define QT4_IMAGEVIEW_QT_H_DEFINED

#include <QtCore/QObject>
#include <QtGui/QTextEdit>
#include <QtGui/qgraphicsview.h>
#include <QtGui/qgraphicsitem.h>

#include "Image/Image.H"
#include "Image/PixelsTypes.H"
#include "Qt4/ImageGraphicsItem.qt.H"

class ImageView : public QGraphicsView
{
  Q_OBJECT

public:
  ImageView(QWidget *parent = 0, const char *name = "unknown");
  virtual ~ImageView();
  void setImage(Image< PixRGB<byte> > img);

protected:
  virtual void wheelEvent(QWheelEvent *event);
  virtual void mousePressEvent(QMouseEvent *event);

  void fitInView();

private:
  ImageGraphicsItem *itsImageDisplay;
  QGraphicsScene *itsScene;

  enum ZoomStatus
  {
    Unknown,
    Fit,
    Zoomed
  } itsZoomStatus;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // QT4_IMAGEVIEW_QT_H_DEFINED
