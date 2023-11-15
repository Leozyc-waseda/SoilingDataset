/*!@file Corba/Parallel/CMap.C a compute a conspicuity map from an image */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/Parallel/CMap.C $
// $Id: CMap.C 14125 2010-10-12 06:29:08Z itti $
//


#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "Image/ColorOps.H"
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
#include <cstdio> // for sprintf

#include "Corba/Parallel/CMapSK.hh"
#include "Corba/ImageOrbUtil.H"

// ######################################################################
// ##### Global options:
// ######################################################################
#define sml        2
#define delta_min  3
#define delta_max  4
#define level_min  0
#define level_max  2
#define maxdepth   (level_max + delta_max + 1)
#define normtyp    (VCXNORM_MAXNORM)

// relative feature weights:
#define IWEIGHT 1.0
#define CWEIGHT 1.0
#define OWEIGHT 1.0
#define FWEIGHT 1.5

#define NBCMAP2 7
//! prescale level by which we downsize images before sending them off
#define PRESCALE 2

CORBA::ORB_var orb;
CosNaming::Name objectName;

static CORBA::Boolean bindObjectToName(CORBA::ORB_ptr, CORBA::Object_ptr);
void unbindObject ();

//Define STAT_CMAP to show camp statisitc and timing
#define STAT_CMAP


//The cmap object implimentation
class CMap_i : public POA_CMap ,
        public PortableServer::RefCountServantBase
{
public:
        inline CMap_i(){}
        virtual ~CMap_i(){}
        virtual ImageOrb* computeCMAP(const ImageOrb& img,
                const short ptyp, const float ori, const float coeff);
        virtual ImageOrb* computeCMAP2(const ImageOrb& img1, const ImageOrb& img2,
                const short ptyp, const float ori, const float coeff);
        virtual ImageOrb* computeFlickerCMAP(const ImageOrb& img,
                const short ptyp, const float ori, const float coeff);

        virtual void shutdown();

private:
        Image<float>* computeCMAP(const Image<float>& fimg,
                                          const PyramidType ptyp, const float ori, const float coeff);
};

void CMap_i::shutdown() {
        // Shutdown the ORB
        unbindObject();
        orb->shutdown(0);
}

ImageOrb* CMap_i::computeCMAP(const ImageOrb& img, const short ptyp, const float ori, const float coeff){


#ifdef STAT_CMAP
                static double avrtime = 0;
                static int avgn = 0;
                double time_taken;
                Timer time(1000000);
                time.reset();
#endif


        //convert to an image
        Image<byte> image;
        orb2Image(img, image);

        //compute the cmap
        Image<float> fimg = image;
        Image<float> *cmap = computeCMAP(fimg, (PyramidType)ptyp, ori, coeff);
        Image<byte> cmap_byte = *cmap;

        delete cmap;

#ifdef STAT_CMAP
                time_taken = time.getSecs();
                avrtime += time_taken;
                ++avgn;
                LINFO("CMap stats: agvn=%i avgtime=%0.4f time_taken=%f ptype=%i ori=%f coeff=%f",
                                avgn, avrtime/(double)avgn, time_taken, (int)ptyp, ori, coeff);
#endif

        return image2Orb(cmap_byte);
}


ImageOrb* CMap_i::computeCMAP2(const ImageOrb& img1, const ImageOrb& img2,
                        const short ptyp, const float ori, const float coeff){

#ifdef STAT_CMAP
                static double avrtime = 0;
                static int avgn = 0;
                double time_taken;

                Timer time(1000000);
                time.reset();
#endif
        //convert to an  image
        Image<byte> image1;
        orb2Image(img1, image1);

        Image<byte> image2;
        orb2Image(img2, image2);

        //compute the cmap
        Image<float> fimg = image1-image2;

        Image<float> *cmap = computeCMAP(fimg, (PyramidType)ptyp, ori, coeff);
        Image<byte> cmap_byte = *cmap;
        delete cmap;

#ifdef STAT_CMAP
                time_taken = time.getSecs();
                avrtime += time_taken;
                ++avgn;
                LINFO("CMap2 stats: agvn=%i avgtime=%0.4f time_taken=%f ptype=%i ori=%f coeff=%f",
                                avgn, avrtime/(double)avgn, time_taken, (int)ptyp, ori, coeff);
#endif
        return image2Orb(cmap_byte);
}

ImageOrb* CMap_i::computeFlickerCMAP(const ImageOrb& img, const short ptyp, const float ori, const float coeff){

#ifdef STAT_CMAP
                static double avrtime = 0;
                static int avgn = 0;
                double time_taken;
                Timer time(1000000);
                time.reset();
#endif
        static Image<float> previmg;

        //convert to an  image
        Image<byte> image;
        orb2Image(img, image);

        //compute the cmap
        Image<float> fimg = image;
        if (previmg.initialized() == false) previmg = fimg;

        previmg -= fimg;
        Image<float> *cmap = computeCMAP(previmg, (PyramidType)ptyp, ori, coeff);
        previmg = fimg;
        Image<byte> cmap_byte = *cmap;

        delete cmap;

#ifdef STAT_CMAP
                time_taken = time.getSecs();
                avrtime += time_taken;
                ++avgn;
                LINFO("FilckerCmap stats: agvn=%i avgtime=%0.4f time_taken=%f ptype=%i ori=%f coeff=%f",
                                avgn, avrtime/(double)avgn, time_taken, (int)ptyp, ori, coeff);
#endif
        return image2Orb(cmap_byte);
}


// ######################################################################
Image<float> *CMap_i::computeCMAP(const Image<float>& fimg, const PyramidType ptyp, const float ori, const float coeff)
{
  // compute pyramid:
  ImageSet<float> pyr = buildPyrGeneric(fimg, 0, maxdepth,
                                        ptyp, ori);

  // alloc conspicuity map and clear it:
  Image<float> *cmap = new Image<float>(pyr[sml].getDims(), ZEROS);


  // intensities is the max-normalized weighted sum of IntensCS:
  for (int delta = delta_min; delta <= delta_max; delta ++)
    for (int lev = level_min; lev <= level_max; lev ++)
      {
        Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
        tmp = downSize(tmp, cmap->getWidth(), cmap->getHeight());
        inplaceAddBGnoise(tmp, 255.0);
        tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
        *cmap += tmp;
      }
  if (normtyp == VCXNORM_MAXNORM)
    *cmap = maxNormalize(*cmap, MAXNORMMIN, MAXNORMMAX, normtyp);
  else
    *cmap = maxNormalize(*cmap, 0.0f, 0.0f, normtyp);

  // multiply by conspicuity coefficient:
  *cmap *= coeff;

  return cmap;
}


//start the class server
int main(int argc, char **argv){

        LINFO("Running as a daemon");
        //Become a daemon
        // fork off the parent process
        pid_t pid = fork();
        if (pid < 0){
                LFATAL("Can not fork");
        }

        if (pid > 0){
                exit(0); //exit the parent process
        }

        // Chnage the file mask
        umask(0);

        //Create a new system id so that the kernel wont think we are an orphan.

        pid_t sid = setsid();

        if (sid < 0){
                LFATAL("Can not become independent");
        }

        //Create the object and run in
        orb = CORBA::ORB_init(argc, argv);

        CORBA::Object_var obj = orb->resolve_initial_references("RootPOA");
        PortableServer::POA_var poa = PortableServer::POA::_narrow(obj);

        CMap_i* cmap = new CMap_i();

        PortableServer::ObjectId_var cmapid = poa->activate_object(cmap);

        //get a ref string
        obj = cmap->_this();
        CORBA::String_var sior(orb->object_to_string(obj));
        std::cerr << "'" << (char*)sior << "'" << "\n";

        if( !bindObjectToName(orb, obj) )
      return 1;
        cmap->_remove_ref();

        PortableServer::POAManager_var pman = poa->the_POAManager();
        pman->activate();



        //run the object untill shutdown or killed
        orb->run();

        LINFO("Shutting down");
        return 0;
}

//////////////////////////////////////////////////////////////////////

static CORBA::Boolean
bindObjectToName(CORBA::ORB_ptr orb, CORBA::Object_ptr objref)
{
  CosNaming::NamingContext_var rootContext;

  try {
    // Obtain a reference to the root context of the Name service:
    CORBA::Object_var obj;
    obj = orb->resolve_initial_references("NameService");

    if( CORBA::is_nil(obj) ) {
      std::cerr << "Obj is null." << std::endl;
      return 0;
    }

    // Narrow the reference returned.
    rootContext = CosNaming::NamingContext::_narrow(obj);
    if( CORBA::is_nil(rootContext) ) {
      std::cerr << "Failed to narrow the root naming context." << std::endl;
      return 0;
    }
  }
  catch(CORBA::ORB::InvalidName& ex) {
    // This should not happen!
    std::cerr << "Service required is invalid [does not exist]." << std::endl;
    return 0;
  }

  try {
    // Bind a context called "test" to the root context:

    CosNaming::Name contextName;
    contextName.length(1);
    contextName[0].id   = (const char*) "test";       // string copied
    contextName[0].kind = (const char*) "saliency"; // string copied
    // Note on kind: The kind field is used to indicate the type
    // of the object. This is to avoid conventions such as that used
    // by files (name.type -- e.g. test.ps = postscript etc.)
    CosNaming::NamingContext_var testContext;
    try {
      // Bind the context to root.
      testContext = rootContext->bind_new_context(contextName);
    }
    catch(CosNaming::NamingContext::AlreadyBound& ex) {
      // If the context already exists, this exception will be raised.
      // In this case, just resolve the name and assign testContext
      // to the object returned:
      CORBA::Object_var obj;
      obj = rootContext->resolve(contextName);
      testContext = CosNaming::NamingContext::_narrow(obj);
      if( CORBA::is_nil(testContext) ) {
        std::cerr << "Failed to narrow naming context." << std::endl;
        return 0;
      }
    }

    // Bind objref with name Echo to the testContext:
    objectName.length(1);


         bool bound = false;
         char CmapID[100];
         for (int i=0; i<100 && !bound; i++) {
                sprintf(CmapID, "CMap_%i", i);
                std::cout << "Binding object " << CmapID << std::endl;
            objectName[0].id   = (const char *) CmapID;   // string copied
            objectName[0].kind = (const char*) "Object"; // string copied

                bound = true;
                try {
                        testContext->bind(objectName, objref);
                }
                catch(CosNaming::NamingContext::AlreadyBound& ex) {
                        //testContext->rebind(objectName, objref);
                        bound = false;
                }

         }
         if (!bound){
                LFATAL("Can not bind object");
                return 0;
         } else {
         }


    // Amendment: When using OrbixNames, it is necessary to first try bind
    // and then rebind, as rebind on it's own will throw a NotFoundexception if
    // the Name has not already been bound. [This is incorrect behaviour -
    // it should just bind].
  }
  catch(CORBA::COMM_FAILURE& ex) {
    std::cerr << "Caught system exception COMM_FAILURE -- unable to contact the "
         << "naming service." << std::endl;
    return 0;
  }
  catch(CORBA::SystemException&) {
    std::cerr << "Caught a CORBA::SystemException while using the naming service."
    << std::endl;
    return 0;
  }

  return 1;
}

// unbind the object from the name server
void unbindObject (){
  CosNaming::NamingContext_var rootContext;

  try {
    // Obtain a reference to the root context of the Name service:
    CORBA::Object_var obj;
    obj = orb->resolve_initial_references("NameService");

    if( CORBA::is_nil(obj) ) {
      std::cerr << "Obj is null." << std::endl;
                return;
    }

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

  CosNaming::Name contextName;
  contextName.length(2);
  contextName[0].id   = (const char*) "test";       // string copied
  contextName[0].kind = (const char*) "saliency"; // string copied
  contextName[1] = objectName[0];

  rootContext->unbind(contextName);
}



