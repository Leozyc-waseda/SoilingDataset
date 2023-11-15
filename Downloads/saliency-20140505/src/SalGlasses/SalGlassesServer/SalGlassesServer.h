/*!@file SalGlassesServer.h control salGlasses via corba  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SalGlasses/SalGlassesServer/SalGlassesServer.h $
// $Id: SalGlassesServer.h 9108 2007-12-30 06:14:30Z rjpeters $
//

#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>

#include "Envision/env_alloc.h"
#include "Envision/env_c_math_ops.h"
#include "Envision/env_image.h"
#include "Envision/env_image_ops.h"
#include "Envision/env_log.h"
#include "Envision/env_mt_visual_cortex.h"
#include "Envision/env_params.h"
#include "Envision/env_pthread_interface.h"
#include "Envision/env_stdio_interface.h"
#include "Envision/env_visual_cortex.h"

//The cmap object implimentation
class SalGlassesServer : public POA_SalGlasses ,
  public PortableServer::RefCountServantBase
{
  public:
    //object specific functions
    SalGlassesServer(int debug);
    virtual ~SalGlassesServer();

    virtual void init();
    virtual void getImageSensorDims(short &w, short &h, const short i);
    virtual ImageOrb* getImageSensor(const short i);
    virtual ImageOrb* getSaliencyMap(const short i);
    virtual void getWinner(short &x, short &y);
    virtual short getSensorValue(const short i);
    virtual void shutdown();

  private:
    float currentSpeed;
    float currentSteering;
    int itsSerialFd;
    int itsDebug;
    struct env_visual_cortex itsIvc;
    struct env_params itsEnvp;

    void sendDriveCommand();
};

