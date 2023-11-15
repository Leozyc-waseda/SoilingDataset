/*!@file Apps/BorderWatch/BorderWatchQt.qt.C Simple GUI for BorderWatch */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Apps/BorderWatch/BorderWatchQt.qt.C $
// $Id: BorderWatchQt.qt.C 15478 2013-08-29 21:14:56Z itti $
//

#include "Apps/BorderWatch/BorderWatchQt.qt.H"

#include <QtCore/QTimer>
#include <QtGui/QLabel>
#include <QtGui/QVBoxLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QListWidget>
#include <QtGui/QSplitter>
#include <QtGui/QFrame>
#include <QtGui/QProgressBar>
#include <QtGui/QLineEdit>
#include <QtGui/QCheckBox>

#include "Apps/BorderWatch/BorderWatchData.H"
#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"
#include "QtUtil/ImageConvert4.H"
#include "Raster/GenericFrame.H"
#include "Util/log.H"
#include "Util/sformat.H"

#include <cstdio>

// ######################################################################
BorderWatchQt::BorderWatchQt(std::vector<GenericFrame>& frames, std::vector<BorderWatchData>& data, QWidget* parent) :
  QWidget(parent), itsThreshold(4.0e-10F), itsFrames(frames), itsLogData(data), itsListIndex(0),
  itsMovieFrame(0), itsZoomed(false), itsPaused(false)
{
  QVBoxLayout *main = new QVBoxLayout(this);
  main->setSpacing(4);
  main->setMargin(2);

  QSplitter* splitter = new QSplitter(Qt::Horizontal, this);
  splitter->setChildrenCollapsible(false);

  itsListWidget = new QListWidget(this);
  itsListWidget->setMinimumWidth(180);

  splitter->addWidget(itsListWidget);
  connect(itsListWidget, SIGNAL(currentRowChanged(int)), this, SLOT(listChanged(int)));

  QFrame *frame = new QFrame(this);
  frame->setFrameShape(QFrame::StyledPanel);

  QVBoxLayout *panel = new QVBoxLayout;

  QHBoxLayout *ed = new QHBoxLayout;
  ed->addStretch(1);

  QLabel *lbl = new QLabel("Threshold:", this);
  ed->addWidget(lbl);

  itsThreshEdit = new QLineEdit(sformat("%g", itsThreshold).c_str(), this);
  ed->addWidget(itsThreshEdit);
  connect(itsThreshEdit, SIGNAL(editingFinished()), this, SLOT(threshChanged()));

  ed->addStretch(1);

  QLabel *lbl2 = new QLabel("       Zoom X2:", this);
  ed->addWidget(lbl2);

  QCheckBox *chk = new QCheckBox(this);
  chk->setCheckState(Qt::Checked); itsZoomed = true;
  connect(chk, SIGNAL(stateChanged(int)), this, SLOT(zoomChanged(int)));
  ed->addWidget(chk);

  ed->addStretch(1);

  panel->addLayout(ed);

  QFrame* hline = new QFrame(this);
  hline->setFrameShape(QFrame::HLine);
  hline->setFrameShadow(QFrame::Raised);
  hline->setLineWidth(2);
  panel->addWidget(hline);

  panel->addStretch(1);

  QHBoxLayout* himage = new QHBoxLayout;
  himage->addStretch(1);
  itsFrameWidget = new QLabel(this);
  himage->addWidget(itsFrameWidget);
  himage->addStretch(1);

  panel->addLayout(himage);

  panel->addStretch(1);

  itsProgressBar = new QProgressBar(this);
  panel->addWidget(itsProgressBar);

  panel->addStretch(1);

  frame->setLayout(panel);
  splitter->addWidget(frame);

  splitter->setStretchFactor(0, 1);
  splitter->setStretchFactor(1, 7);  // preferentially stretch the image pane over the text list pane

  splitter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  main->addWidget(splitter);

  itsStatusLabel = new QLabel("Status: Idle.", this);
  itsStatusLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
  main->addWidget(itsStatusLabel);

  this->setLayout(main);

  // populate the list with a bunch of events:
  parseEvents();

  // get our timer going:
  itsTimer = new QTimer(this);
  itsTimer->setInterval(33); // in milliseconds
  connect(itsTimer, SIGNAL(timeout()), this, SLOT(timerTick()));
  itsTimer->start();

  // Receive keyboard events:
  setFocusPolicy(Qt::ClickFocus);
}

// ######################################################################
BorderWatchQt::~BorderWatchQt()
{
  itsTimer->stop();
}

// ######################################################################
void BorderWatchQt::listChanged(const int idx)
{
  itsListIndex = idx;
  if (itsEvents.size()) {
    const uint estart = itsEvents[idx].start, eend = itsEvents[idx].end;

    itsListWidget->setCurrentRow(itsListIndex);
    itsMovieFrame = estart;
    itsProgressBar->setRange(estart, eend - 1);
    itsProgressBar->setValue(estart);

    // compute max score achieved over this event:
    float mscore = 0.0F;
    for (uint i = estart; i < eend; ++i) mscore = std::max(mscore, itsLogData[i].score);

    itsStatusLabel->setText(sformat("Event %d || Frames: %06u - %06u || Max Score: %g", idx,
                                    estart, eend-1, mscore).c_str());
  }
}

// ######################################################################
void BorderWatchQt::parseEvents()
{
  itsEvents.clear(); itsListWidget->clear(); bool inevent = false; const uint margin = 33; uint i = 0;
  char buf[100]; uint count = 0;
  std::vector<BorderWatchData>::const_iterator itr = itsLogData.begin(), stop = itsLogData.end();
  BWevent e;
  while (itr != stop) {
    // are we above threshold?
    if (itr->score >= itsThreshold) {
      count = 0; // then continue any ongoing event
      // start a new event?
      if (inevent == false)
        { inevent = true; e.start = i; if (e.start <= margin) e.start = 0; else e.start -= margin; }
    }

    // write out a completed event?
    if (inevent && count > margin) {
      inevent = false; count = 0; e.end = i;
      itsEvents.push_back(e);
      snprintf(buf, 100, "%04" ZU ": %06u-%06u", itsEvents.size(), e.start, e.end);
      itsListWidget->addItem(buf);
    }
    ++itr; ++i; ++count;
  }

  // maybe one last event is still open:
  if (inevent) {
    e.end = itsLogData.size()-1;
    itsEvents.push_back(e);
    snprintf(buf, 100, "%04" ZU ": %06u-%06u", itsEvents.size(), e.start, e.end);
    itsListWidget->addItem(buf);
  }

  // reset our position in our list:
  listChanged(0);

  snprintf(buf, 100, "Extracted %" ZU " events above threshold = %e", itsEvents.size(), itsThreshold);
  itsStatusLabel->setText(buf);
}

// ######################################################################
void BorderWatchQt::timerTick()
{
  if (itsEvents.size())
  {
    // get the current movie frame and log data:
    GenericFrame genframe = itsFrames[itsMovieFrame];
    Image<PixRGB<byte> > im = genframe.asRgbU8();
    if (itsZoomed) im = quickInterpolate(im, 2);
    BorderWatchData& d = itsLogData[itsMovieFrame];

    // add some drawings:
    Point2D<int> p = d.salpoint; if (itsZoomed) { p.i *= 2; p.j *= 2; }
    if (d.score > itsThreshold) drawCircle(im, p, itsZoomed ? 30 : 15, PixRGB<byte>(255,255,0), 2);
    else drawCircle(im, p, itsZoomed ? 10 : 5, PixRGB<byte>(0,128,0), 1);
    writeText(im, Point2D<int>(10,0), sformat("%s - S=%g %06d", d.itime.c_str(), d.score, d.iframe).c_str(),
              PixRGB<byte>(255, 64, 0), PixRGB<byte>(0), SimpleFont::FIXED(6), true);

    // display the image in our widget:
    QPixmap pixmap = convertToQPixmap4(im);
    itsFrameWidget->setPixmap(pixmap);

    // update progress bar:
    itsProgressBar->setValue(itsMovieFrame);

    // roll on to the next frame, if not paused:
    if (itsPaused == false)
    {
      ++itsMovieFrame;
      if (itsMovieFrame >= itsEvents[itsListIndex].end) itsMovieFrame = itsEvents[itsListIndex].start;
    }
  }
}

// ######################################################################
void BorderWatchQt::threshChanged()
{
  QString txt = itsThreshEdit->text(); bool ok;
  float t = txt.toFloat(&ok);

  if (ok) { itsThreshold = t; parseEvents(); }
  else itsStatusLabel->setText("Invalid threshold value");
}

// ######################################################################
void BorderWatchQt::zoomChanged(int state)
{
  if (state == Qt::Checked) itsZoomed = true; else itsZoomed = false;
}
// ######################################################################
void BorderWatchQt::keyPressEvent(QKeyEvent *event)
{
  if (event->key() == Qt::Key_Space) itsPaused = ! itsPaused;
  QWidget::keyPressEvent(event);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
