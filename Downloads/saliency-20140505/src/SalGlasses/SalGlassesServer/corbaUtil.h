/*!@file CorbaUtil.h */

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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SalGlasses/SalGlassesServer/corbaUtil.h $
// $Id: corbaUtil.h 9108 2007-12-30 06:14:30Z rjpeters $
//

#ifndef CORBAUTIL_H
#define CORBAUTIL_H

#include <omniORB4/Naming.hh>
#include <omniORB4/omniURI.h>

OMNI_USING_NAMESPACE(omni);


//! Build an object reference for many object within a context
bool getMultiObjectRef(CORBA::ORB_ptr orb, const char *contextName, CORBA::Object_ptr *dst, int &nObj);


//! Register with the name service
CORBA::Boolean
bindObjectToName(CORBA::ORB_ptr orb, CORBA::Object_ptr objref,
                 const char* contextId, const char* contextKind,
                 const char* objectPrefix, CosNaming::Name& objectName);

// unbind the object from the name server
void unbindObject (CORBA::ORB_ptr orb, const char* contextId,
                          const char* contextKind,
                          CosNaming::Name &objectName);

#endif

