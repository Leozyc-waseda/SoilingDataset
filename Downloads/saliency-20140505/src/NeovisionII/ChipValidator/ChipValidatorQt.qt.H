/*!@file NeovisionII/ChipValidator/ChipValidatorQt.qt.H Simple GUI to validate chips */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Apps/BorderWatch/ChipValidatorQt.qt.H $
// $Id: ChipValidatorQt.qt.H 13059 2010-03-26 08:14:32Z itti $
//

#ifndef NEOVISIONII_CHIPVALIDATORQT_QT_H_DEFINED
#define NEOVISIONII_CHIPVALIDATORQT_QT_H_DEFINED

#include <QtGui/QWidget>
#include <QtGui/QLabel>
#include <QtGui/QMouseEvent>
class QGridLayout;
class QApplication;

#include <vector>
#include <string>

#include "Image/Image.H"
#include "Image/PixelsTypes.H"

class ChipQLabel; // defined below

// ######################################################################
//! Simple struct for each chip
struct ChipData {
  Image<PixRGB<byte> > image;
  std::string file;
  bool positive;
};

// ######################################################################
//! Simple GUI to display and browse BorderWatch results
class ChipValidatorQt : public QWidget
{
  Q_OBJECT

public:
  //! Constructor
  ChipValidatorQt(QApplication *qapp, std::vector<ChipData>& chipvec, const Dims& griddims, QWidget* parent = 0);

  //! Destructor
  virtual ~ChipValidatorQt();

  //! A chip was clicked
  void chipClicked(const uint idx);

protected slots:
  void pageChanged(const int idx);

private:
  std::vector<ChipData>& itsChipVec;
  Dims itsGridDims;
  std::vector<ChipQLabel*> itsChipLabels;
  uint itsPage;
  QLabel *itsStatusLabel;

  //! display an image in a chip
  void setChipImage(const uint idx);
};


// ######################################################################
//! a clickable QLabel
class ChipQLabel : public QLabel {

  Q_OBJECT

public:
  ChipQLabel(const uint grididx, ChipValidatorQt *master, Qt::WindowFlags flags = 0) :
    QLabel(master, flags), itsGridIdx(grididx), itsMaster(master)
  { setMouseTracking(true); }

  virtual ~ChipQLabel() { }

  //! Override to catch button presses
  virtual void mousePressEvent(QMouseEvent *eve)
  { if (eve->button() == Qt::LeftButton) itsMaster->chipClicked(itsGridIdx); }

private:
  const uint itsGridIdx;
  ChipValidatorQt *itsMaster;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // NEOVISIONII_CHIPVALIDATORQT_QT_H_DEFINED
