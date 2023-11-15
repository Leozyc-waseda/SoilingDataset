/*!@file Qt4/ImageView.qt.C */

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
// $HeadURL$ svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt4/ImageView.qt.C $
//

#include <QtGui/QtGui>
#include "Qt4/ImageView.qt.H"
#include "QtUtil/ImageConvert4.H"

// ######################################################################
ImageView::ImageView(QWidget *parent, const char* name) :
  QGraphicsView(parent),
  itsZoomStatus(Unknown)
{
  setAttribute(Qt::WA_DeleteOnClose);

  //Create a new graphics scene which will handle all of our objects
  itsScene = new QGraphicsScene;
  setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);

  //Create a new display widget to show our movie images
  itsImageDisplay = new ImageGraphicsItem;
  itsScene->addItem(itsImageDisplay);
  itsImageDisplay->setZValue(1);

  QGraphicsTextItem *text = new QGraphicsTextItem(QString(name));
  itsScene->addItem(text);
  text->setZValue(-1);

  //Set itsScene as the main scene for this viewer
  setScene(itsScene);
}

// ######################################################################
ImageView::~ImageView()
{
}

// ######################################################################
void ImageView::setImage(Image< PixRGB<byte> > img)
{
  itsImageDisplay->setImage(convertToQImage4(img));

  if (itsZoomStatus == Unknown)
  {
    fitInView();
  }
  else
  {
    itsImageDisplay->update();
    update();
  }
}

// ######################################################################
void ImageView::fitInView()
{
  QRectF rect = itsImageDisplay->getRect();

  if (rect.height() <= 1e-3 || rect.width() <= 1e-3) return;

  // set the image viewer to fit the image in the view
  itsZoomStatus = Fit;
  setSceneRect(itsImageDisplay->getRect());
  QGraphicsView::fitInView(itsImageDisplay, Qt::KeepAspectRatio);
  itsImageDisplay->update();
  update();
}

// ######################################################################
void ImageView::wheelEvent(QWheelEvent *event)
{
  // call the base class to perform default behaviour
  QGraphicsView::wheelEvent(event);

  // ignore the event if the zoom state is unknown
  if (itsZoomStatus == Unknown) return;

  // ignore if already zoomed too much
  double currentZoom = matrix().determinant();
  currentZoom = sqrt(currentZoom > 0 ? currentZoom : -currentZoom);

  if (event->delta() < 0 && currentZoom < 0.1) return;
  if (event->delta() > 0 && currentZoom > 4.0) return;

  // let's zoom the image accordingly
  const double delta = static_cast<double>(event->delta()) / 120.0;
  const double sc = (delta > 0) ? (delta * 1.25) : (-delta / 1.25);
  scale(sc, sc);
  itsZoomStatus = Zoomed;
  itsImageDisplay->update();
  update();
  event->accept();
}

// ######################################################################
void ImageView::mousePressEvent(QMouseEvent *event)
{
  // call the base class to perform default behaviour
  QGraphicsView::mousePressEvent(event);

  // ignore the event if the zoom state is unknown
  if (itsZoomStatus == Unknown) return;

  if ((event->buttons() & Qt::MidButton) != 0)
  {
    fitInView();
    event->accept();
    return;
  }

  // TODO: Add easy panning of the image by mouse dragging
/*if ((event->buttons() & Qt::LeftButton) != 0 && itsZoomStatus == Zoomed)
  {
    // scroll the image
    event->accept();
    return;
  }*/
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

