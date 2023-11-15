/*!@file Corba/Objects/SceneRecServer.H control a SceneRec via corba  */

//////////////////////////////////////////////////////////////////// //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/Objects/SceneRecServer.H $
// $Id: SceneRecServer.H 7087 2006-08-31 15:57:53Z rjpeters $
//

#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include "Image/ColorOps.H"
#include "Image/MathOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Util/Assert.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Component/ModelManager.H"
#include "Corba/Objects/SceneRecServerSK.hh"
#include "Corba/ImageOrbUtil.H"
#include "RCBot/SceneRec.H"


//The cmap object implimentation
class SceneRecServer_i : public POA_SceneRecServer ,
        public PortableServer::RefCountServantBase,
        public SceneRec
{
        public:
                //object specific functions
                SceneRecServer_i(OptionManager& mgr);
                virtual ~SceneRecServer_i();

                //! Setup up with image to be processed
                void newInput(const ImageOrb &img);

                //! Is the last given input done processing?
                short outputReady();

                //! Get the feature point. Used for landmark navigation
                short getLandmarkLoc(Point2DOrb &loc);

                //! Train the system on a given feature location
                void trainFeature(const ImageOrb &img, const Point2DOrb &loc,
                                const DimsOrb &window, const short leg);

                virtual void shutdown();

};

