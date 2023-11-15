/*!@file Corba/Parallel/GetSM.C a compute a saliency map from cmap class */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/Parallel/GetSM.C $
// $Id: GetSM.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWinManaged.H"
#include "GUI/XWindow.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/PyramidTypes.H"
#include "Image/ShapeOps.H"     // for decX() etc.
#include "Image/Transforms.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/SaccadeControllers.H"
#include "Psycho/PsychoDisplay.H"
#include "Transport/FrameIstream.H"
#include "Util/Assert.H"
#include "Util/Timer.H"
#include <iostream>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>


#include "Corba/ImageOrbUtil.H"
#include "Corba/Parallel/CMapSK.hh"
#include <omniORB4/Naming.hh>
#include <omniORB4/omniURI.h>

OMNI_USING_NAMESPACE(omni);

void getObjectReference(CORBA::ORB_ptr orb);

XWinManaged window(Dims(256, 256), -1, -1, "Test Output");

#define sml        2
#define delta_min  3
#define delta_max  4
#define level_min  0
#define level_max  2
#define maxdepth   (level_max + delta_max + 1)
#define normtyp    (VCXNORM_MAXNORM)

#define IWEIGHT 1.0
#define CWEIGHT 1.0
#define OWEIGHT 1.0
#define FWEIGHT 1.5

#define NBCMAP2 8
//! prescale level by which we downsize images before sending them off
#define PRESCALE 2

//!Number of cmap objects that we have running
int nCmapObj =0;

CMap_var CMap_ref[100];

//!The current cmap object to send the request to
CMap_var getCmapRef(){
        static int current_obj = 0;

        //just do a round rubin
        current_obj  = (current_obj+1)%nCmapObj;

        LINFO("Using cmap object number %i\n", current_obj);
        return CMap_ref[current_obj];
}

Image<float> cmap[NBCMAP2];


class CmapThread : public omni_thread {

        public:
                CmapThread(Image<byte> &img, int mapid, PyramidType ptype, double ori, double weight){
                        th_img1 = img; th_mapid = mapid; th_ptype = ptype; th_ori = ori; th_weight = weight;
                        LINFO("Starting thread for cmap[%i] with %i %g %g\n",mapid, ptype, ori, weight);

                        start_undetached();
                        //start();
                }

                CmapThread(Image<byte> &img1, Image<byte> &img2, int mapid, PyramidType ptype, double ori, double weight){
                        th_img1 = img1; th_img2 = img2; th_mapid = mapid; th_ptype = ptype; th_ori = ori; th_weight = weight;
                        LINFO("Starting thread for cmap[%i] with %i %g %g\n",mapid, ptype, ori, weight);

                        start_undetached();
                        //start();
                }

                ~CmapThread(void) {}

   private:
                Image<byte> th_img1;
                Image<byte> th_img2;
                int th_mapid;
                PyramidType th_ptype;
                double th_ori;
                double th_weight;

                void* run_undetached(void *ptr){
                        //process the thread
                        CMap_var CMap = getCmapRef();        //get the object to send to
                        ImageOrb* imgOrb;
                        switch (th_mapid){
                                case 5:
                                        imgOrb = CMap->computeCMAP2(*image2Orb(th_img1), *image2Orb(th_img2),
                                                                                                                 th_ptype, th_ori, th_weight);
                                        break;
                                case 6:
                                        imgOrb = CMap->computeCMAP2(*image2Orb(th_img1), *image2Orb(th_img2),
                                                                                                                 th_ptype, th_ori, th_weight);
                                        break;
                                case 7:
                                        imgOrb = CMap->computeFlickerCMAP(*image2Orb(th_img1),
                                                                                                                                 th_ptype, th_ori, th_weight);
                                        break;

                                default:
                                        imgOrb = CMap->computeCMAP(*image2Orb(th_img1),
                                                                                                           th_ptype, th_ori, th_weight);
                        }

                        Image<byte> tmp;
                        orb2Image(*imgOrb, tmp);
                        cmap[th_mapid] = tmp;
                        delete imgOrb;
                        LINFO("Cmap %i done\n",th_mapid);

                        return NULL;
                }

};


Image< PixRGB<byte> > &getImage(){

        int w=256, h=256;
        static Image< PixRGB<byte> > *img =0;

        if (img) delete img;


        LINFO("New Image\n");
        img = new Image< PixRGB<byte> >(w, h, NO_INIT);


        //generate a random image

        Image<byte> r(w,h,ZEROS), g(w,h,ZEROS), b(w,h,ZEROS);

        //r.speckleNoise(1.0F, 10, 255, true);
        //g.speckleNoise(1.0F, 10, 255, true);
        //b.speckleNoise(1.0F, 10, 255, true);

        *img = makeRGB(r,g,b);

        int x_pos = 1+(int)((float)w*rand()/(RAND_MAX+1.0));
        int y_pos = 1+(int)((float)h*rand()/(RAND_MAX+1.0));
        printf("Position: %i %i\n", x_pos, y_pos);

        //draw a red cross
        drawCross(*img,Point2D<int>(x_pos, y_pos), PixRGB<byte>(255,0,0), 30, 8);
        drawCross(*img,Point2D<int>(x_pos,y_pos), PixRGB<byte>(0,0,255), 20, 4);


        //read an image
        //*img = Raster::ReadRGB("test.ppm"); //read image

        //window2.drawImage(rescale(*img, 256, 256));
        return *img;
}

int main(int argc, char **argv){


        CORBA::ORB_ptr orb = CORBA::ORB_init(argc,argv,"omniORB4");

   getObjectReference(orb);

        Timer time(1000000);
        time.reset();

        Image<float> sm(256>>2, 256>>2, ZEROS);

        for (int ii=0; ii<30; ii++){
        //start the clock

                LINFO("Loop i=%i\n", ii);
                Image< PixRGB<byte> > img;
                img = getImage();

                Image<PixRGB<byte> > ima2 =
            decY(lowPass5y(decX(lowPass5x(img),1<<PRESCALE)),1<<PRESCALE);

                Image<byte> lum;
                lum = luminance(img);


                // compute RG and BY
                Image<byte> rImg, gImg, bImg, yImg;
                getRGBY(img, rImg, gImg, bImg, yImg, (byte)25);


                static class CmapThread* thCmapWorker[NBCMAP2];

                thCmapWorker[0] = new CmapThread(lum, 0, Gaussian5, 0.0, IWEIGHT); //lum thread

                thCmapWorker[1] = new CmapThread(lum, 1, Oriented5, 0.0, OWEIGHT); //ori thread
                thCmapWorker[2] = new CmapThread(lum, 2, Oriented5, 45.0, OWEIGHT); //ori thread
                thCmapWorker[3] = new CmapThread(lum, 3, Oriented5, 90.0, OWEIGHT); //ori thread
                thCmapWorker[4] = new CmapThread(lum, 4, Oriented5, 135.0, OWEIGHT); //ori thread

                thCmapWorker[5] = new CmapThread(rImg, gImg, 5, Gaussian5, 0.0, CWEIGHT); //color thread
                thCmapWorker[6] = new CmapThread(bImg, yImg, 6, Gaussian5, 0.0, CWEIGHT); //color thread

                thCmapWorker[7] = new CmapThread(lum, 7, Gaussian5, 0.0, FWEIGHT); //filker thread


                //wait for threads to finish
                for(int i=0; i< NBCMAP2; i++) {
                        thCmapWorker[i]->join(NULL);
                }

                Image<float> sminput;

                for (int i=0; i< NBCMAP2; i++){
                        if (cmap[i].initialized()){
                                if (sminput.initialized())
                                        sminput += cmap[i];
                                else
                                        sminput = cmap[i];
                        }
                }

                //inhibit old input and add the new one from sminput
                if (sminput.initialized()) sm = sm * 0.7F + sminput * 0.3F;

                // find most salient location and feed saccade controller:
                float maxval; Point2D<int> currwin; findMax(sm,currwin, maxval);

                printf("Winner at %i %i level=%f\n", currwin.i, currwin.j, maxval);

                Image<float> dispsm(sm); inplaceNormalize(dispsm, 0.0F, 255.0F);

                Image<byte> showsm = dispsm;
                drawCircle(showsm, currwin, 5, (byte)255, 1);
                window.drawImage(rescale(showsm, 256, 256));
                //exit(0);

        }

        LINFO("Time taken %f\n", time.getSecs());

        //shutdown all the cmap servers

        for (int i=0; i< nCmapObj; i++){
                CMap_ref[i]->shutdown();
        }
        return 0;

}

void getObjectReference(CORBA::ORB_ptr orb) {
  CosNaming::NamingContext_var rootContext;

  try {
    // Obtain a reference to the root context of the Name service:
    CORBA::Object_var obj;
    obj = orb->resolve_initial_references("NameService");

    // Narrow the reference returned.
    rootContext = CosNaming::NamingContext::_narrow(obj);
    if( CORBA::is_nil(rootContext) ) {
      std::cerr << "Failed to narrow the root naming context." << std::endl;
      return;
    }
  }
  catch(CORBA::ORB::InvalidName& ex) {
    // This should not happen!
    std::cerr << "Service required is invalid [does not exist]." << std::endl;
    return;
  }

  //find out how many objects we have to work with

  // Create a name object, containing the name test/context:
  CosNaming::Name_var name = omniURI::stringToName("test.saliency");

  CORBA::Object_var obj = rootContext->resolve(name);

  CosNaming::NamingContext_var context;
  context = CosNaming::NamingContext::_narrow(obj);

  if (CORBA::is_nil(context)) {
          LINFO("No objects found\n");
          exit(1);
  }

  //get all the objects in the context

  CosNaming::BindingIterator_var bi;
  CosNaming::BindingList_var bl;
  CosNaming::Binding_var b;

  context->list(0, bl, bi);

  if (CORBA::is_nil(bi)){
          LINFO("No objects found\n");
          exit(1);
  }

  nCmapObj=0;
  while (bi->next_one(b)){

          CosNaming::Name name_comp = b->binding_name;
         // CORBA::String_var sname = omniURI::nameToString(b->binding_name);

          LINFO("Binding to %s ... ", (const char*)omniURI::nameToString(name_comp));
          CORBA::Object_ptr ref = context->resolve(name_comp);
          CMap_ref[nCmapObj] = CMap::_narrow(ref);
          //check for errors
          if (!CORBA::is_nil(CMap_ref[nCmapObj])){
                  nCmapObj++;
                  LINFO("Done\n");
          } else {
                  LINFO("Fail\n");
          }

  }
  bi->destroy();

}


