/*!@file Parallel/SingleChannelBeoServer.H server to work with SingleChannelBeo */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/SingleChannelBeoServer.H $
// $Id: SingleChannelBeoServer.H 6988 2006-08-11 17:15:38Z rjpeters $
//

#ifndef SINGLECHANNELBEOSERVER_H_DEFINED
#define SINGLECHANNELBEOSERVER_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Util/Types.H"
#include "rutz/shared_ptr.h"

class Beowulf;
class SingleChannel;

//! Server to work with SingleChannelBeo
/*! This class receives images from SingleChannelBeo, process them,
  and return the pyramids and submaps back. One server may serve
  several SingleChannelBeo clients; messages will be processed
  according to the Beowulf node number of the client. The first thing
  the client should do is to sent a configuration message, according
  to which we will build the channel here. Then, the client may send
  input images, which we will process, and we will return the
  resulting pyramid and maps.

  The system can work in two modes: in full emulation mode, the output
  map, channel pyramid, submaps and clip pyramid are all returned by
  the processor node. This requires a lot of network bandwidth but
  should ensure results that are exactly identical to when using the
  normal channels. In quick mode, only the output submap is
  returned. This will work with standard processing that only deals
  with the output, but things like saving the submaps or biasing the
  visual cortex will break; so use the fast mode at your own risk ;-)
  Selection of which mode is used is done via opur itsQuickMode
  ModelParam.  */

class SingleChannelBeoServer : public ModelComponent
{
public:
  //! Constructor
  SingleChannelBeoServer(OptionManager& mgr,
                         const std::string& descrName =
                         "SingleChannelBeo Server",
                         const std::string& tagName =
                         "SingleChannelBeoServer");

  //! Destructor
  virtual ~SingleChannelBeoServer();

  //! Call this often for us to do our work
  void check();

protected:
  OModelParam<bool> itsQuickMode; //!< only send off output map, not other maps

private:
  // our Beowulf for message passing
  nub::soft_ref<Beowulf> itsBeo;

  // find the channel for given node, or NULL if no channel for it yet:
  nub::soft_ref<SingleChannel> channel(const int32 node);
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
