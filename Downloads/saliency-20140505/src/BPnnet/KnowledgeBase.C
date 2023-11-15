/*!@file BPnnet/KnowledgeBase.C Knowledge Base class */

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
// Primary maintainer for this file: Philip Williams <plw@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BPnnet/KnowledgeBase.C $
// $Id: KnowledgeBase.C 4786 2005-07-04 02:18:56Z itti $
//

#include "BPnnet/KnowledgeBase.H"

#include "Util/log.H"
#include <fstream>

// ######################################################################
KnowledgeBase::KnowledgeBase( void )
{
  // nothing to do
}

// ######################################################################
KnowledgeBase::~KnowledgeBase()
{ }

// ######################################################################
bool KnowledgeBase::load(const char *fname)
{
  std::ifstream s(fname);
  if (s.is_open() == false) { LERROR("Cannot read %s", fname); return false; }
  char buf[256];

  while(!s.eof())
    {
      s.getline(buf, 256);
      if (strlen(buf) > 1)
        {
          SimpleVisualObject vo(buf);
          addSimpleVisualObject(vo);
        }
    }
  s.close();
  return true;
}

// ######################################################################
bool KnowledgeBase::save(const char *fname) const
{
  std::ofstream s(fname);
  if (s.is_open() == false) { LERROR("Cannot write %s", fname); return false; }
  int sz = getSize();
  for (int i = 0; i < sz; i ++) s<<vokb[i].getName()<<std::endl;
  s.close();
  return true;
}

// ######################################################################
bool KnowledgeBase::addSimpleVisualObject( SimpleVisualObject& o )
{
  // check if vo we're trying to add already exists in kb
  // if so, do not add and return false; otherwise, add and return true
  if (findSimpleVisualObjectIndex(o.getName()) != -1) return false;
  else { vokb.push_back(o); return true; }
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
