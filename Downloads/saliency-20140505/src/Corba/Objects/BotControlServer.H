/*!@file Corba/Objects/BotControlServer.H control a robot via corba  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/Objects/BotControlServer.H $
// $Id: BotControlServer.H 13993 2010-09-20 04:54:23Z itti $
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
#include "Corba/Objects/BotControlServerSK.hh"
#include "Corba/ImageOrbUtil.H"
#include "RCBot/BotControl.H"


//The cmap object implimentation
class BotControlServer_i : public POA_BotControlServer ,
                           public PortableServer::RefCountServantBase,
                           public BotControl
{
        public:
                //object specific functions
                BotControlServer_i(OptionManager& mgr, bool debug);
                virtual ~BotControlServer_i();

                virtual void init();
                virtual float getSpeed();
                virtual short setSpeed(const float speed);
                virtual float getSteering();
                virtual short setSteering(const float steeringPos);
                virtual ImageOrb* getImageSensor(const short i);
                virtual void getImageSensorDims(short &w, short &h, const short i);
                void setInfo(const char *info, const Point2DOrb& trackLoc, const Point2DOrb& recLoc);
                short getUserInput(Point2DOrb &loc);

                virtual void shutdown();

};

