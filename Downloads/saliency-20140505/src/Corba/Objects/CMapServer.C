/*!@file Corba/Objects/CMapServer.C a compute a conspicuity map from an image */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/Objects/CMapServer.C $
// $Id: CMapServer.C 9412 2008-03-10 23:10:15Z farhan $
//

#include <stdlib.h>
#include <time.h>
#include <iostream>
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
#include "Corba/Objects/CMapSK.hh"
#include "Corba/ImageOrbUtil.H"
#include "Corba/Objects/CMapServer.H"
#include "Corba/CorbaUtil.H"
//#include "GUI/XWinManaged.H"
//XWinManaged xwin(Dims(256, 256), -1, -1, "CMap Debug");

// ######################################################################
// ##### Global options:
// ######################################################################
#define delta_min  3
#define delta_max  4
#define level_min  0
#define level_max  2
#define maxdepth   (level_max + delta_max + 1)
#define normtyp    (VCXNORM_MAXNORM)
//#define normtyp    (VCXNORM_FANCYWEAK)
//#define normtyp    (VCXNORM_FANCYONE)


#define NBCMAP2 7
//! prescale level by which we downsize images before sending them off
#define PRESCALE 2

CORBA::ORB_var orb;
CosNaming::Name objectName;

bool Debug = false;

#ifndef LocalCMapServer
//! Signal handler (e.g., for control-C)
void terminate(int s)
{
        LERROR("*** INTERRUPT ***");
        unbindObject(orb, "saliency", "CMapServers", objectName);
        orb->shutdown(0);
}
#endif

static omni_mutex BiasCenterSurroundMutex;

class BiasCenterSurroundThread : public omni_thread {

        public:
                BiasCenterSurroundThread(Image<float> &cmap, ImageSet<float> &pyr,
                                int lev1, int lev2, float bias):
                        th_cmap(cmap), th_pyr(pyr), th_lev1(lev1), th_lev2(lev2), th_bias(bias){

                                start_undetached();
                        }

        private:
                Image<float> &th_cmap;
                ImageSet<float> &th_pyr;
                int th_lev1, th_lev2;
                float th_bias;

                void* run_undetached(void *ptr){

                        Image<float> tmp = centerSurround(th_pyr, th_lev1, th_lev2, true);
                        inplaceNormalize(tmp, 0.0F, 255.0F);

                        //bias this level
                        for (Image<float>::iterator itr = tmp.beginw(), stop = tmp.endw();
                                        itr != stop; ++itr) {
                                *itr = 255.0F - fabs((*itr) - th_bias); //corelate the bias
                        }

                        tmp = downSize(tmp, th_cmap.getWidth(), th_cmap.getHeight());
                        //tmp = rescale(tmp, cmap->getWidth(), cmap->getHeight());
                        //inplaceAddBGnoise(tmp, 255.0);
                        tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
                        BiasCenterSurroundMutex.lock();
                        th_cmap += tmp;
                        BiasCenterSurroundMutex.unlock();

                        return NULL;
                }
};

class CenterSurroundThread : public omni_thread {

        public:
                CenterSurroundThread(Image<float> &cmap, ImageSet<float> &pyr, int lev1, int lev2):
                        th_cmap(cmap), th_pyr(pyr), th_lev1(lev1), th_lev2(lev2){

                                start_undetached();
                        }

        private:
                Image<float> &th_cmap;
                ImageSet<float> &th_pyr;
                int th_lev1, th_lev2;

                void* run_undetached(void *ptr){

                        Image<float> tmp = centerSurround(th_pyr, th_lev1, th_lev2, true);
                        inplaceNormalize(tmp, 0.0F, 255.0F);

                        if (tmp.getWidth() >= th_cmap.getWidth())
                                tmp = downSize(tmp, th_cmap.getWidth(), th_cmap.getHeight());
                        else
                                tmp = quickInterpolate(tmp, th_cmap.getWidth()/tmp.getWidth());

                        tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);

                        BiasCenterSurroundMutex.lock();
                        th_cmap += tmp;
                        BiasCenterSurroundMutex.unlock();

                        return NULL;
                }
};

void CMap_i::shutdown() {
        // Shutdown the ORB
        unbindObject(orb,"saliency", "CMapServers", objectName);
        orb->shutdown(0);
}

void CMap_i::setSaliencyMapLevel(const short sml){
        saliencyMapLevel = sml;
}



// ######################################################################
CMap::BiasSeq* CMap_i::getBiasCMAP(const ImageOrb& img, const short ptyp,
                const float ori, const float coeff,
                const Point2DOrb& loc)
{
        //convert to an image
        Image<byte> image;
        orb2Image(img, image);

        //compute the cmap
        Image<float> fimg = image;
        // compute pyramid:
        ImageSet<float> pyr = buildPyrGeneric(fimg, 0, maxdepth,
                                              (PyramidType)ptyp, ori);


        // alloc conspicuity map and clear it:

        CMap::BiasSeq *bias = new CMap::BiasSeq;
        bias->length(6); //TODO: compute from delta_min etc..

        // intensities is the max-normalized weighted sum of IntensCS:
        int ii=0;
        for (int delta = delta_min; delta <= delta_max; delta ++)
                for (int lev = level_min; lev <= level_max; lev ++)
                {
                        Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
                        inplaceNormalize(tmp, 0.0F, 255.0F);

                        if (loc.i >= 0 && loc.j >=0){
                                //get the bias points
                                //Map the pooint to the current point in the pyramid level
                                Point2D<int> newp((int)((float)loc.i * ((float)tmp.getWidth() / (float)fimg.getWidth())),
                                                (int)((float)loc.j * ((float)tmp.getHeight() / (float)fimg.getHeight())) );
                                (*bias)[ii] = (float)tmp.getVal(newp);
                                //tmp.setVal(newp, 255);
                                //xwin.drawImage(rescale(tmp, 256, 256));
                                //sleep(2);

                        }

                        ii++;
                }


        return bias;
}

ImageOrb* CMap_i::computeCMAP(const ImageOrb& img, const short ptyp, const float ori, const float coeff){



        static double avrtime = 0;
        static int avgn = 0;
        double time_taken;
        Timer time(1000000);
        if (Debug) {
                time.reset();
        }

        //convert to an image
        Image<byte> image;
        orb2Image(img, image);
        Image<float> fimg = image;

        // compute pyramid:
        ImageSet<float> pyr = buildPyrGeneric(fimg, 0, maxdepth,
                                              (PyramidType)ptyp, ori);

        // alloc conspicuity map and clear it:
        Image<float> cmap(pyr[saliencyMapLevel].getDims(), ZEROS);

        int ii=0;

        CenterSurroundThread *workth[25]; //TODO: should be computed

        // intensities is the max-normalized weighted sum of IntensCS:
        for (int delta = delta_min; delta <= delta_max; delta ++)
                for (int lev = level_min; lev <= level_max; lev ++)
                {
                        //spawn a new thread for each centersurround computation
                        workth[ii] = new CenterSurroundThread(cmap,pyr, lev, lev + delta);
                        ii++;
                }

        //wait for thread to finish
        for (int i=0; i<ii; i++)
                workth[i]->join(NULL);

        // inplaceAddBGnoise(cmap, 25.0F);

        if (normtyp == VCXNORM_MAXNORM)
                cmap = maxNormalize(cmap, MAXNORMMIN, MAXNORMMAX, normtyp);
        else
                cmap = maxNormalize(cmap, 0.0f, 0.0f, normtyp);

        // multiply by conspicuity coefficient:
        if (coeff != 1.0F) cmap *= coeff;

        if (Debug) {
                time_taken = time.getSecs();
                avrtime += time_taken;
                ++avgn;
                LINFO("CMap stats: agvn=%i avgtime=%0.4f time_taken=%f ptype=%i ori=%f coeff=%f",
                                avgn, avrtime/(double)avgn, time_taken, (int)ptyp, ori, coeff);
        }

        return image2Orb(cmap);


}


ImageOrb* CMap_i::computeCMAP2(const ImageOrb& img1, const ImageOrb& img2,
                const short ptyp, const float ori, const float coeff){

        static double avrtime = 0;
        static int avgn = 0;
        double time_taken;
        Timer time(1000000);
        if (Debug) {

                time.reset();
        }
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

        if (Debug) {
                time_taken = time.getSecs();
                avrtime += time_taken;
                ++avgn;
                LINFO("CMap2 stats: agvn=%i avgtime=%0.4f time_taken=%f ptype=%i ori=%f coeff=%f",
                                avgn, avrtime/(double)avgn, time_taken, (int)ptyp, ori, coeff);
        }
        return image2Orb(cmap_byte);
}

ImageOrb* CMap_i::computeFlickerCMAP(const ImageOrb& img, const short ptyp, const float ori, const float coeff){

        static double avrtime = 0;
        static int avgn = 0;
        double time_taken;
        Timer time(1000000);
        if (Debug) {
                time.reset();
        }
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

        if (Debug) {
                time_taken = time.getSecs();
                avrtime += time_taken;
                ++avgn;
                LINFO("FilckerCmap stats: agvn=%i avgtime=%0.4f time_taken=%f ptype=%i ori=%f coeff=%f",
                                avgn, avrtime/(double)avgn, time_taken, (int)ptyp, ori, coeff);
        }

        return image2Orb(cmap_byte);
}

ImageOrb* CMap_i::computeBiasCMAP(const ImageOrb& img, const short ptyp,
                const float ori, const float coeff,
                const CMap::BiasSeq& bias){

        static double avrtime = 0;
        static int avgn = 0;
        double time_taken;
        Timer time(1000000);
        if (Debug) {
                time.reset();
        }

        //convert to an image
        Image<byte> image;
        orb2Image(img, image);
        Image<float> fimg = image;

        // compute pyramid:
        ImageSet<float> pyr = buildPyrGeneric(fimg, 0, maxdepth,
                                              (PyramidType)ptyp, ori);

        // alloc conspicuity map and clear it:
        Image<float> cmap(pyr[saliencyMapLevel].getDims(), ZEROS);

        if (Debug) {
                LINFO("Bias map with: ");
                for (uint i=0; i<bias.length(); i++)
                        printf("%f ", bias[i]);
                printf("\n");
        }


        // intensities is the max-normalized weighted sum of IntensCS:
        int ii=0;

        BiasCenterSurroundThread *workth[25]; //TODO: should be computed

        for (int delta = delta_min; delta <= delta_max; delta ++)
                for (int lev = level_min; lev <= level_max; lev ++)
                {
                        /*     Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
                                                 inplaceNormalize(tmp, 0.0F, 255.0F);

                        //bias this level
                        for (Image<float>::iterator itr = tmp.beginw(), stop = tmp.endw();
                        itr != stop; ++itr) {
                         *itr = 255.0F - fabs((*itr) - bias[ii]); //corelate the bias
                         }

                         tmp = downSize(tmp, cmap.getWidth(), cmap.getHeight());
                        //tmp = rescale(tmp, cmap->getWidth(), cmap->getHeight());
                        //inplaceAddBGnoise(tmp, 255.0);
                        tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
                        cmap += tmp;
                        ii++; */

                        //spawn a new thread for each centersurround computation
                        workth[ii] = new BiasCenterSurroundThread(cmap,pyr, lev, lev + delta, bias[ii]);
                        ii++;

                }

        //wait for thread to finish
        for (int i=0; i<ii; i++)
                workth[i]->join(NULL);

        if (normtyp == VCXNORM_MAXNORM)
                cmap = maxNormalize(cmap, MAXNORMMIN, MAXNORMMAX, normtyp);
        else
                cmap = maxNormalize(cmap, 0.0f, 0.0f, normtyp);

        // multiply by conspicuity coefficient:
        if (coeff != 1.0F) cmap *= coeff;

        if (Debug) {
                time_taken = time.getSecs();
                avrtime += time_taken;
                ++avgn;
                LINFO("CMap stats: agvn=%i avgtime=%0.4f time_taken=%f ptype=%i ori=%f coeff=%f",
                                avgn, avrtime/(double)avgn, time_taken, (int)ptyp, ori, coeff);
        }

        return image2Orb(cmap);
}


// ######################################################################
Image<float> *CMap_i::computeCMAP(const Image<float>& fimg, const PyramidType ptyp, const float ori, const float coeff)
{

        // compute pyramid:
        ImageSet<float> pyr = buildPyrGeneric(fimg, 0, maxdepth,
                                              ptyp, ori);

        // alloc conspicuity map and clear it:
        Image<float> *cmap = new Image<float>(pyr[saliencyMapLevel].getDims(), ZEROS);


        // intensities is the max-normalized weighted sum of IntensCS:
        for (int delta = delta_min; delta <= delta_max; delta ++)
                for (int lev = level_min; lev <= level_max; lev ++)
                {
                        Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
                        tmp = downSize(tmp, cmap->getWidth(), cmap->getHeight());
                        //tmp = rescale(tmp, cmap->getWidth(), cmap->getHeight());
                        inplaceAddBGnoise(tmp, 255.0);
                        tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
                        *cmap += tmp;
                }
        if (normtyp == VCXNORM_MAXNORM)
                *cmap = maxNormalize(*cmap, MAXNORMMIN, MAXNORMMAX, normtyp);
        else
                *cmap = maxNormalize(*cmap, 0.0f, 0.0f, normtyp);

        // multiply by conspicuity coefficient:
        if (coeff != 1.0F) *cmap *= coeff;


        return cmap;
}

// ######################################################################
Image<float> *CMap_i::computeBiasCMAP(const Image<float>& fimg, const PyramidType ptyp,
                const float ori, const float coeff,
                const CMap::BiasSeq& bias)
{
        // compute pyramid:
        ImageSet<float> pyr = buildPyrGeneric(fimg, 0, maxdepth,
                                              ptyp, ori);

        // alloc conspicuity map and clear it:
        Image<float> *cmap = new Image<float>(pyr[saliencyMapLevel].getDims(), ZEROS);

        if (Debug) {
                LINFO("Bias map with: ");
                for (uint i=0; i<bias.length(); i++)
                        printf("%f ", bias[i]);
                printf("\n");
        }


        // intensities is the max-normalized weighted sum of IntensCS:
        int ii=0;
        for (int delta = delta_min; delta <= delta_max; delta ++)
                for (int lev = level_min; lev <= level_max; lev ++)
                {
                        Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
                        inplaceNormalize(tmp, 0.0F, 255.0F);

                        //bias this level
                        for (Image<float>::iterator itr = tmp.beginw(), stop = tmp.endw();
                                        itr != stop; ++itr) {
                                *itr = 255.0F - fabs((*itr) - bias[ii]); //corelate the bias
                        }

                        tmp = downSize(tmp, cmap->getWidth(), cmap->getHeight());
                        //tmp = rescale(tmp, cmap->getWidth(), cmap->getHeight());
                        //inplaceAddBGnoise(tmp, 255.0);
                        tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
                        *cmap += tmp;
                        ii++;
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

        MYLOGVERB = LOG_INFO;

        // Instantiate a ModelManager:
        ModelManager manager("CMap Object");

        if (manager.parseCommandLine((const int)argc, (const char**)argv, "", 0, 0) == false)
                return(1);


        if (manager.debugMode()){
                Debug = true;
        } else {
                LINFO("Running as a daemon. Set --debug to see any errors.");
                //Become a daemon
                // fork off the parent process
                pid_t pid = fork();
                if (pid < 0)
                        LFATAL("Can not fork");

                if (pid > 0)
                        exit(0); //exit the parent process

                // Change the file mask
                umask(0);

                //Create a new system id so that the kernel wont think we are an orphan.
                pid_t sid = setsid();
                if (sid < 0)
                        LFATAL("Can not become independent");

                fclose(stdin);
                fclose(stdout);
                fclose(stderr);

        }
        // catch signals and redirect them to terminate for clean exit:
        signal(SIGHUP, terminate); signal(SIGINT, terminate);
        signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
        signal(SIGALRM, terminate);

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

        if( !bindObjectToName(orb, obj, "saliency", "CMapServers", "CMap", objectName) ){
                LFATAL("Can not bind to name service");
                return 1;
        }
        cmap->_remove_ref();

        PortableServer::POAManager_var pman = poa->the_POAManager();
        pman->activate();

        try {
                manager.start();
                //run the object untill shutdown or killed
                orb->run();
        } catch (...) {
                LINFO("Error starting server");
        }

        LINFO("Shutting down");
        unbindObject(orb, "saliency", "CMapServers", objectName);

        manager.stop();
        return 0;
}

