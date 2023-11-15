/*!@file Qt4/ImageDisplayLayout.qt.C */

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
// $HeadURL$ svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt4/ImageDisplayLayout.qt.C $
//

#include <QtGui/QtGui>
#include "Qt4/ImageDisplayLayout.qt.H"
#include "Qt4/ImageView.qt.H"

using namespace std;

// ######################################################################
ImageDisplayLayout::ImageDisplayLayout(QWidget *parent, const std::string& layout) :
  QWidget(parent)
{
  // erase the white spaces first
  string layoutString;
  const string delimiter(" \t\r\n");
  for (string::const_iterator itr = layout.begin(); itr != layout.end(); ++itr)
  {
    if (delimiter.find(*itr) == string::npos)
      layoutString += *itr;
  }
  QGridLayout *grid = CreateLayout(layoutString.begin(), layoutString.end());
  if (grid != NULL)
  {
    setLayout(grid);
  }
}

// ######################################################################
ImageDisplayLayout::~ImageDisplayLayout()
{
}

// ######################################################################
QGridLayout *ImageDisplayLayout::CreateLayout(string::const_iterator begin, string::const_iterator end)
{
  if (begin == end) return NULL;

  int row = 0;
  int col = 0;
  QGridLayout *grid = new QGridLayout;

  for (string::const_iterator itr = begin; itr != end;)
  {
    if (*itr == '(')
    {
      int parenthese_count = 1;
      for (string::const_iterator itr2 = itr + 1; itr2 != end; ++itr2)
      {
        if (*itr2 == '(')
        {
          parenthese_count++;
        }
        else if (*itr2 == ')')
        {
          parenthese_count--;
          if (parenthese_count == 0)
          {
            QGridLayout *grid2 = CreateLayout(itr + 1, itr2);
            if (grid2 != NULL)
            {
              grid->addLayout(grid2, row, col);
            }
            itr = itr2 + 1;
            break;
          }
        }
      }
      if (parenthese_count != 0) LFATAL("Unbalanced parentheses detected in layout string");
      if (itr == end) break;
    }
    if (*itr == ',')
    {
      col++;
      ++itr;
    }
    else if(*itr == ';')
    {
      col = 0;
      row++;
      ++itr;
    }
    else
    {
      string::const_iterator itr2 = itr;
      string name;
      for (; itr2 != end && *itr2 != ',' && *itr2 != ';' && *itr2 != '(' && *itr2 != ')'; ++itr2)
      {
        name += *itr2;
      }
      ImageView *view = new ImageView(0, name.c_str());
      grid->addWidget(view, row, col);
      itsViews.insert(pair<string,ImageView*>(name, view));
      itr = itr2;
    }
  }
  return grid;
}

// ######################################################################
ImageView* ImageDisplayLayout::view(const char* name)
{
  map<std::string, ImageView*>::iterator item = itsViews.find(string(name));
  return (item == itsViews.end()) ? NULL : item->second;
}

// ######################################################################
void ImageDisplayLayout::setImage(const char* name, Image< PixRGB<byte> > img)
{
  ImageView* v = view(name);
  if (v == NULL) LFATAL("Accessing a NULL or not existing view '%s'", name);
  v->setImage(img);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

