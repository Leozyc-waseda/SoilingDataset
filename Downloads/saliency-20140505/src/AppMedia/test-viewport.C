/*!@file AppMedia/test-viewport.C test the opengl viewport */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/test-viewport.C $
// $Id: test-viewport.C 12962 2010-03-06 02:13:53Z irock $


#include "GUI/ViewPort.H"
#include "Util/log.H"
#include "Util/WorkThreadServer.H"
#include "Util/JobWithSemaphore.H"
#include "Component/ModelManager.H"
#include "Raster/GenericFrame.H"
#include "Image/Layout.H"
#include "Image/MatrixOps.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include <stdlib.h>
#include <math.h>

double pos[3];
double side[3];
double R[12];

Image<PixRGB<byte> > frame;

namespace
{
  class ViewPortLoop : public JobWithSemaphore
  {
    public:
      ViewPortLoop()
        :
          itsPriority(1),
          itsJobType("controllerLoop")
    {}

      virtual ~ViewPortLoop() {}

      virtual void run()
      {
        double cameraParam[3][4] = {
          {350.475735, 0, 158.250000, 0},
          {0.000000, -363.047091, 118.250000, 0.000000},
          {0.000000, 0.000000, 1.000000, 0.00000}};

        ViewPort vp("ViewPort", "ground.ppm", "", false, false,
            320, 240, cameraParam);
        vp.setTextures(false);
        vp.setDrawWorld(false);
        vp.setWireframe(false);
        vp.setZoom(0.85);

        //Set the camera
        double cam_xyz[3] = {10.569220,-210,140.959999};
        double cam_hpr[3] = {102.500000,-26.500000,-5.500000};
        vp.dsSetViewpoint (cam_xyz, cam_hpr);


        //ViewPort::DSObject object = vp.load3DSObject("./etc/spaceship.3ds", "./etc/textures/spaceshiptexture.ppm");
        //object.scale = 0.01;

        while(1)
        {
          double cameraParam[16]  = {
            0.995441, 0.044209, -0.084521, 0.000000, 0.095269, -0.504408, 0.858193, 0.000000,
            -0.004694, -0.862333, -0.506320, 0.000000, -6.439873, 11.187120, 197.639489, 1.000000};

          vp.initFrame(cameraParam);
          vp.dsDrawSphere(pos,R,1.3f);
          //vp.dsSetTexture (ViewPort::WOOD);
          vp.dsSetColor (1,1,0.0);

          vp.dsSetColor (0.5,0.5,0.5);
          vp.dsDrawBox(pos,R,side);
          //vp.dsSetTexture (ViewPort::OTHER, &objectTexture);
          //vp.dsDraw3DSObject(pos, R, object);

          vp.updateFrame();

          frame = flipVertic(vp.getFrame());

        }
      }

      virtual const char* jobType() const
      { return itsJobType.c_str(); }

      virtual int priority() const
      { return itsPriority; }

    private:
      const int itsPriority;
      const std::string itsJobType;
  };
}

int main(int argc, char *argv[]){

  ModelManager manager("Test Viewport");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);
  // let's get all our ModelComponent instances started:
        manager.start();


  pos[0] = 0;
  pos[1] = 0;
  pos[2] = 5;
  R[0] = 1; R[1] = 0; R[2] = 0;
  R[4] = 0;  R[5] = 1; R[6] = 0;
  R[8] = 0; R[9] = 0; R[10] = 1;

  side[0] = 30; side[1] = 30; side[2] = 30;


  rutz::shared_ptr<WorkThreadServer> itsThreadServer;
  itsThreadServer.reset(new WorkThreadServer("ViewPort",1)); //start a single worker thread
  itsThreadServer->setFlushBeforeStopping(false);
  rutz::shared_ptr<ViewPortLoop> j(new ViewPortLoop());
  itsThreadServer->enqueueJob(j);


  frame = Image<PixRGB<byte> >(255, 255, ZEROS);
        int run=1;
        while(run){
    ofs->writeRGB(frame, "viewport", FrameInfo("ViewPort", SRC_POS));
    usleep(1000);
         }

  exit(0);

}


