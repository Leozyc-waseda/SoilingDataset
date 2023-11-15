/*!@file Qt4/ImageDisplayLayout.qt.H */

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
// $HeadURL$ svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt4/ImageDisplayLayout.qt.H $
//

#ifndef QT4_IMAGEDISPLAYLAYOUT_QT_H_DEFINED
#define QT4_IMAGEDISPLAYLAYOUT_QT_H_DEFINED

#include <map>
#include <string>

#include <QtCore/QObject>
#include <QtGui/QWidget>
#include <QtGui/qgridlayout.h>

#include "Image/Image.H"
#include "Image/PixelsTypes.H"

class ImageView;

//! Create image display layout using Qt
/*! Layout sctring specifies how how to layout the ImageView objects on widget.
    ImageView objects can be accessed by the name to display images and superimposed
    vectors.
    e.g. N1           => creates a single ImageView object:  [N1]
         N1,N2        => creates a single row containing two ImageView objects: [N1 | N2]
         N1;N2        => creates a single column containing two ImageView objects: [N1
                                                                                    N2]
         N1,N2;N3,N4  => creates 2x2 grid layout with four ImageView objects: [N1 | N2
                                                                               N3 | N4]
         N1,N2;N3,(X1;X2)
                      => creates a 2x2 grid layout with an embedded 2x1 grid as last element:
                                                                              [N1 | N2
                                                                               N3 | X1
                                                                                  | X2]
                        (N3 has the same height as (X1;X2) element)
*/
class ImageDisplayLayout : public QWidget
{
  Q_OBJECT

public:
  ImageDisplayLayout(QWidget *parent = 0, const std::string& layout = "");
  virtual ~ImageDisplayLayout();

  ImageView* view(const char* name);
  void setImage(const char* name, Image< PixRGB<byte> > img);

private:
  QGridLayout *CreateLayout(std::string::const_iterator begin, std::string::const_iterator end);

private:
  std::map<std::string, ImageView*> itsViews;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // QT4_IMAGEDISPLAYLAYOUT_QT_H_DEFINED
