/*!@file Apps/BorderWatch/BorderWatchQt.qt.H Simple GUI for BorderWatch */

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
// Primary maintainer for this file: Laurent Itti
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Apps/BorderWatch/BorderWatchQt.qt.H $
// $Id: BorderWatchQt.qt.H 15478 2013-08-29 21:14:56Z itti $
//

#ifndef APPS_BORDERWATCH_BORDERWATCHQT_QT_H_DEFINED
#define APPS_BORDERWATCH_BORDERWATCHQT_QT_H_DEFINED

#include <QtGui/QWidget>
#include <QtGui/QKeyEvent>
class QListWidget;
class QLabel;
class GenericFrame;
class BorderWatchData;
class QProgressBar;
class QLineEdit;
#include <vector>

//! Simple GUI to display and browse BorderWatch results
class BorderWatchQt : public QWidget
{
  Q_OBJECT

public:
  //! Constructor
  BorderWatchQt(std::vector<GenericFrame>& frames, std::vector<BorderWatchData>& data, QWidget* parent = 0);

  //! Destructor
  virtual ~BorderWatchQt();


protected slots:
  void listChanged(const int idx);
  void timerTick();
  void threshChanged();
  void zoomChanged(int state);

protected:
  void keyPressEvent(QKeyEvent *event);

private:
  QListWidget *itsListWidget;
  QLabel *itsFrameWidget;
  QLabel *itsStatusLabel;
  QProgressBar *itsProgressBar;
  QTimer *itsTimer;
  QLineEdit *itsThreshEdit;
  float itsThreshold;
  std::vector<GenericFrame>& itsFrames;
  std::vector<BorderWatchData>& itsLogData;
  uint itsListIndex;
  uint itsMovieFrame;
  bool itsZoomed;
  bool itsPaused;

  struct BWevent { uint start; uint end; };
  std::vector<BWevent> itsEvents;

  void parseEvents();
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPS_BORDERWATCH_BORDERWATCHQT_QT_H_DEFINED
