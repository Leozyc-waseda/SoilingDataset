/*!@file NeovisionII/ChipValidator/ChipValidatorQt.qt.C Simple GUI to validate chips */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Apps/BorderWatch/ChipValidatorQt.qt.C $
// $Id: ChipValidatorQt.qt.C 13059 2010-03-26 08:14:32Z itti $
//

#include "NeovisionII/ChipValidator/ChipValidatorQt.qt.H"

#include <QtGui/QVBoxLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QSlider>
#include <QtGui/QGridLayout>
#include <QtGui/QPushButton>
#include <QtGui/QApplication>

#include "Image/DrawOps.H"
#include "QtUtil/ImageConvert4.H"
#include "Util/log.H"
#include "Util/sformat.H"

// ######################################################################
ChipValidatorQt::ChipValidatorQt(QApplication *qapp, std::vector<ChipData>& chipvec,
                                 const Dims& griddims, QWidget* parent) :
  QWidget(parent), itsChipVec(chipvec), itsGridDims(griddims), itsChipLabels(), itsPage(0)
{
  QVBoxLayout *main = new QVBoxLayout(this);
  main->setSpacing(4);
  main->setMargin(2);

  QHBoxLayout *hgrid = new QHBoxLayout;
  hgrid->addStretch(1);

  QGridLayout *grid = new QGridLayout;
  grid->setSpacing(3); uint idx = 0;
  for (int j = 0; j < itsGridDims.h(); ++j)
    for (int i = 0; i < itsGridDims.w(); ++i) {
      ChipQLabel *cl = new ChipQLabel(idx, this);
      grid->addWidget(cl, j, i);
      itsChipLabels.push_back(cl);
      setChipImage(idx);
      ++idx;
    }
  hgrid->addLayout(grid);

  hgrid->addStretch(1);

  main->addLayout(hgrid);

  QHBoxLayout *hcontrol = new QHBoxLayout;

  QSlider *slider = new QSlider(Qt::Horizontal , this);
  slider->setRange(0, itsChipVec.size() / itsGridDims.sz() + ( (itsChipVec.size() % itsGridDims.sz() ) ? 0 : -1));
  slider->setValue(0);
  slider->setPageStep(1);
  slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
  hcontrol->addWidget(slider);
  connect(slider, SIGNAL(valueChanged(int)), this, SLOT(pageChanged(int)));

  QPushButton *button = new QPushButton("Save + Exit", this);
  hcontrol->addWidget(button);
  connect(button, SIGNAL(pressed()), qapp, SLOT(quit()));

  main->addLayout(hcontrol);

  itsStatusLabel = new QLabel("Status: Idle.", this);
  itsStatusLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
  main->addWidget(itsStatusLabel);

  this->setLayout(main);
}

// ######################################################################
ChipValidatorQt::~ChipValidatorQt()
{ }

// ######################################################################
void ChipValidatorQt::setChipImage(const uint idx)
{
  if (itsChipVec.size() == 0) LFATAL("No chips loaded, I need at least one chip to work");
  if (idx >= uint(itsGridDims.sz())) LFATAL("Trying to set chip image for out-of-grid chip");

  // get the image, if it exists
  const uint fullidx = itsPage * itsGridDims.sz() + idx;

  if (fullidx < itsChipVec.size()) {
    Image< PixRGB<byte> > im = itsChipVec[fullidx].image;

    // mark chip as negative?
    if (itsChipVec[fullidx].positive == false) {
      drawLine(im, Point2D<int>(0, 0), Point2D<int>(im.getWidth()-1, im.getHeight()-1), PixRGB<byte>(255, 0, 0), 2);
      drawLine(im, Point2D<int>(im.getWidth()-1, 0), Point2D<int>(0, im.getHeight()-1), PixRGB<byte>(255, 0, 0), 2);
    }

    // draw the image:
    QPixmap pixmap = convertToQPixmap4(im);
    itsChipLabels[idx]->setPixmap(pixmap);
  } else {
    // set an empty pixmap:
    Image<PixRGB<byte> > im(itsChipVec[0].image.getDims(), NO_INIT); im.clear(PixRGB<byte>(192));
    QPixmap pixmap = convertToQPixmap4(im);
    itsChipLabels[idx]->setPixmap(pixmap);
  }
}

// ######################################################################
void ChipValidatorQt::chipClicked(const uint idx)
{
  if (idx >= uint(itsGridDims.sz())) LFATAL("Clicked an out-of-grid chip");

  const uint fullidx = itsPage * itsGridDims.sz() + idx;
  if (fullidx < itsChipVec.size()) {
    itsChipVec[fullidx].positive = ! itsChipVec[fullidx].positive;
    itsStatusLabel->setText(sformat("Chip %u/%" ZU " marked as %s.", fullidx, itsChipVec.size()-1,
                                    itsChipVec[fullidx].positive ? "positive" : "negative").c_str());
    setChipImage(idx);
  }
}

// ######################################################################
void ChipValidatorQt::pageChanged(const int idx)
{
  itsPage = idx;

  const uint startidx = itsPage*itsGridDims.sz();
  const uint stopidx = std::min(startidx + itsGridDims.sz() - 1, uint(itsChipVec.size() - 1));
  itsStatusLabel->setText(sformat("Showing chips %u - %u", startidx, stopidx).c_str());

  for (uint i = 0; i < uint(itsGridDims.sz()); ++i) setChipImage(i);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
