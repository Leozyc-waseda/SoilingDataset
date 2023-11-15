/*!@file BPnnet/KnowledgeBase.H Header file for knowledge base */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BPnnet/KnowledgeBase.H $
// $Id: KnowledgeBase.H 9308 2008-02-22 19:04:41Z rjpeters $
//

#ifndef KNOWLEDGEBASE_H_DEFINED
#define KNOWLEDGEBASE_H_DEFINED

#include "BPnnet/SimpleVisualObject.H"

#include <cstring>
#include <vector>

//! this class holds a knowledge base
class KnowledgeBase {
public:
  //! Constructor
  KnowledgeBase();

  //! Destructor
  ~KnowledgeBase();

  //! load from disk
  bool load(const char *fname);

  //! save to disk
  bool save(const char *fname) const;

  //! Add new object to knowledge base. True on success.
  bool addSimpleVisualObject(SimpleVisualObject& o);

  //! Find visual object from KB that has corresponding name
  /*! returns true on success */
  inline bool findSimpleVisualObject(const char *name, SimpleVisualObject& o) const;

  //! Find visual object indexed by "index" in the kb array
  /*! returns true on success */
  inline bool findSimpleVisualObject(const int index, SimpleVisualObject& o) const;

  //! Return visual object from KB that has corresponding name
  /*! or -1 of object not found */
  inline int findSimpleVisualObjectIndex(const char *name) const;

  //! get visual object at given index:
  inline SimpleVisualObject getSimpleVisualObject(const int index) const;

  //! Return number of visual objects held in this knowledge base
  inline int getSize() const;

private:
  std::vector<SimpleVisualObject> vokb;
};

// ######################################################################
// ########## INLINED METHODS:
// ######################################################################

inline bool KnowledgeBase::findSimpleVisualObject(const char *name,
                                            SimpleVisualObject& o) const
{
  int index = findSimpleVisualObjectIndex(name);
  if (index == -1) { o = SimpleVisualObject("<UNKNOWN>"); return false; }
  else { o = vokb[index]; return true; }
}

// ######################################################################
inline bool KnowledgeBase::findSimpleVisualObject(const int index,
                                            SimpleVisualObject& o) const
{
  if (index < 0 || index >= getSize())
    { o = SimpleVisualObject("<UNKNOWN>"); return false; }
  else { o = vokb[index]; return true; }
}

// ######################################################################
inline int KnowledgeBase::findSimpleVisualObjectIndex(const char *name) const
{
  int n = getSize();
  for (int i = 0; i < n; i ++)
    if (strcmp(name, vokb[i].getName()) == 0) return i;
  return -1;
}

// ######################################################################
inline SimpleVisualObject KnowledgeBase::getSimpleVisualObject(const int index) const
{
  if (index < 0 || index >= getSize()) return SimpleVisualObject("<UNKNOWN>");
  else return vokb[index];
}

// ######################################################################
inline int KnowledgeBase::getSize() const
{ return (int)(vokb.size()); }

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
