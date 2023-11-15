/*!@file Corba/Objects/CMap.H */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/Objects/CMap.H $
// $Id: CMap.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef CMAP_H
#define CMAP_H

#include "Corba/Objects/CMapSK.hh"
#include "Corba/ImageOrbUtil.H"
#include "Corba/CorbaUtil.H"
#include "Image/PyramidTypes.H"

static omni_mutex CMapWorkerThreadMutex;

//the worker cmap threads
class CMapWorkerThread : public omni_thread {

        public:
                CMapWorkerThread(CMap_ptr cmap, Image<float> &dst, Image<byte> &img,  PyramidType ptype,
                                double ori, double weight, int &frame, bool isUndetached) :  th_dst(dst), th_bias(NULL),
                                th_ptype(ptype), th_ori(ori), th_weight(weight),  th_frame(frame)  {

                                        CMapObj = cmap;
                                        th_img1 = img; //copy the image incase it changes

                                        LDEBUG("Starting thread for with %i %g %g\n", ptype, ori, weight);

                                        runBias = false;
                                        if (isUndetached)
                                                start_undetached();
                                        else
                                                start();
                }

                CMapWorkerThread(CMap_ptr cmap, Image<float> &dst, Image<byte> &img,  PyramidType ptype,
                                double ori, double weight, int &frame, std::vector<float> &bias, bool isUndetached) :
                                th_dst(dst), th_ptype(ptype), th_ori(ori), th_weight(weight), th_frame(frame) {

                                        CMapObj = cmap;
                                        th_img1 = img; //copy the image incase it changes

                                        th_bias = new CMap::BiasSeq;
                                        th_bias->length(bias.size());
                                        for(uint i=0; i<bias.size(); i++)
                                                (*th_bias)[i] = bias[i];


                                        LDEBUG("Starting thread for with %i %g %g\n", ptype, ori, weight);

                                        runBias = true;
                                        if (isUndetached)
                                                start_undetached();
                                        else
                                                start();
                }

                ~CMapWorkerThread(){}

   private:
                Image<byte> th_img1;
                Image<byte> th_img2;
                Image<float> &th_dst;
                CMap::BiasSeq *th_bias;

                PyramidType th_ptype;
                double th_ori;
                double th_weight;
                int &th_frame;
                CMap_ptr CMapObj;
                bool runBias;

                void* run_undetached(void *ptr){
                        LDEBUG("Start undetached");
                        processCMap();

                        return NULL;
                }

                void run(void *ptr){
                        LDEBUG("Start detached");
                        processCMap();
                }

                void processCMap (){
                        //process the thread

                        ImageOrb *imgOrb;

                        if (runBias){
                                LDEBUG("Biasing cmap");


                                imgOrb = CMapObj->computeBiasCMAP(*image2Orb(th_img1),
                                                th_ptype, th_ori, th_weight, *th_bias);
                        } else {
                                LDEBUG("Unbias CMap");
                                imgOrb = CMapObj->computeCMAP(*image2Orb(th_img1),
                                                th_ptype, th_ori, th_weight);
                        }


                        Image<byte> tmp;
                        orb2Image(*imgOrb, tmp);
                        delete imgOrb;

                        //here we assign the results to the image givien. Need to insure we dont
                        //conflict with another thred writing to the same image space
                        CMapWorkerThreadMutex.lock();
                        th_dst = tmp; //convert to float
                        CMapWorkerThreadMutex.unlock();
                        th_frame++; //update how many frames have we proceesd
                }


};

// The main thread manegemnt class the spawns the worker threads
class CMapThreads {
        public:
                CMapThreads(CORBA::ORB_ptr orb)
                {
                        nCmapObj = 0;
                        initObject(orb);
                }

                //start a new thread and return the thread address
                //if isUndetrached is true then Join needs to be called
                CMapWorkerThread* newth(Image<float> &dst, Image<byte> &img, PyramidType ptype,
                                                                          double ori, double weight, int &frame, bool isUndetached = false)
                {
                        CMap_ptr CMap = getCmapRef();        //get the object to send to
                        return new CMapWorkerThread(CMap, dst, img, ptype, ori, weight,frame, isUndetached);
                }

                CMapWorkerThread* newth(Image<float> &dst, Image<byte> &img, PyramidType ptype,
                                                                          double ori, double weight, int &frame, std::vector<float> &bias,
                                                                          bool isUndetached = false)
                {
                        CMap_ptr CMap = getCmapRef();        //get the object to send to
                        return new CMapWorkerThread(CMap, dst, img, ptype, ori, weight,frame, bias, isUndetached);
                }

                std::vector<float> &getBias(Image<float> &dst, Image<byte> &img, PyramidType ptype,
                                                                          double ori, double weight, int &frame, Point2D<int> &loc)
                {
                        Point2DOrb locOrb;
                        locOrb.i = loc.i; locOrb.j = loc.j;

                        CMap_ptr CMap = getCmapRef();        //get the object to send to
                        CMap::BiasSeq *bias = CMap->getBiasCMAP(*image2Orb(img),
                                                                                                           ptype, ori, weight, locOrb);
                        std::vector<float> *retBias = new std::vector<float>(bias->length());

                        for (uint i=0; i<bias->length(); i++){
                                (*retBias)[i] = (*bias)[i];
                        }

                        return *retBias;

                }

        private:

                //!Number of cmap objects that we have running
                int nCmapObj;
                CORBA::Object_ptr CMap_ref[100]; //100 is the limit for now TODO: watch for overflow


                //!The current cmap object to send the request to
                CMap_var getCmapRef(){
                        static int current_obj = 0;

                        //just do a round robin
                        current_obj  = (current_obj+1)%nCmapObj;

                        LDEBUG("Using cmap object number %i\n", current_obj);

                        return CMap::_narrow(CMap_ref[current_obj]);
                }

                void initObject(CORBA::ORB_ptr orb) {
                        LDEBUG("Initalizing CMap object");
                        if (!getMultiObjectRef(orb, "test.saliency", CMap_ref, nCmapObj)){
                                LFATAL("Can not find any object to bind with");
                        }

                }
};

#endif

