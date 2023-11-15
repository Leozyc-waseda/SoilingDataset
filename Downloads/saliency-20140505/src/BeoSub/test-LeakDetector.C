/*!@file BeoSub/test-LeakDetector.C Leak Detector for BeoSub */

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
// Primary maintainer for this file
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-LeakDetector.C $
// $Id: test-LeakDetector.C 6990 2006-08-11 18:13:51Z rjpeters $
//

// - Syamsul


#include "Devices/BeoChip.H"
#include "BeoSub/BeoSub.H"


//just because of the virtual funcs
class BeoSubLeakDetector : public BeoSub {
        public:
        BeoSubLeakDetector(OptionManager& mgr,const std::string& descrName,const std::string& tagName);
        virtual void turnOpen(Angle, bool);
        virtual void advanceRel(float f, bool);
        virtual Image<PixRGB<byte> > grabImage(BeoSubCamera) const ;
        virtual void dropMarker(bool md);
};

//leak detector class
class LeakDetector : public ModelComponent {

        public:
                LeakDetector(OptionManager& mgr,
                                const std::string& descrName="LeakDetector",
                                const std::string& tagName="LeakDetector");
                void setBeoChip(nub::soft_ref<BeoChip>& bc);
                void setBeoSub(nub::soft_ref<BeoSubLeakDetector>& sub);
                bool getState() {return leakState;}

                ~LeakDetector();

        protected:
                void start1();
                void stop1();

                //get reference from beochip

        private:
                nub::soft_ref<BeoChip> itsBeoChip;
                nub::soft_ref<BeoSubLeakDetector> itsBeoSub;
                void dispatchBeoChipEvent(BeoChipEventType t, int valint, float valfloat);
                bool leakState;

        friend class LeakDetectorListener;

};


//listener for Leak Detector
class LeakDetectorListener :public BeoChipListener{
        public:
                LeakDetectorListener(nub::soft_ref<LeakDetector>& ld) : itsLeakDetector(ld) {}

                void event(const BeoChipEventType t, const int valint, const float valfloat) {
                        LDEBUG("Event type %d , vint : %d , vfloat : %f", int(t),valint,valfloat);
                        itsLeakDetector->dispatchBeoChipEvent(t,valint,valfloat);

                }

        private:
                nub::soft_ref<LeakDetector> itsLeakDetector;

};


//leak detector implementation
LeakDetector::LeakDetector(OptionManager& mgr,const std::string& descrName,const std::string& tagName):
                                ModelComponent(mgr,descrName,tagName)
{
}

LeakDetector::~LeakDetector() {
        //pthread_mutex_destroy(&itsMutex);
}

void LeakDetector::start1() {
        //start thread to listen to leak detector sensor
        //itsKeepGoing = true;
        //pthread_mutex_create(&itsRunner,NULL,LeakDetector_run,(void*)this);
        return;
}

void LeakDetector::stop1() {
        //itsKeepGoing = false;
        usleep(300000); //thread exiting.
        return;

}

void LeakDetector::setBeoChip(nub::soft_ref<BeoChip>& bc) {
        itsBeoChip = bc;

}

void LeakDetector::setBeoSub(nub::soft_ref<BeoSubLeakDetector>& sub) {

        itsBeoSub = sub ;
}

void LeakDetector::dispatchBeoChipEvent(BeoChipEventType t, int valint, float valfloat) {

        LDEBUG("Event : %d, valint : %d valfloat : %f",int(t),valint, valfloat);

        //sub do something to kill yourself.
        return;

}

BeoSubLeakDetector::BeoSubLeakDetector(OptionManager& mgr, const std::string& descrName,const std::string& tagName):
                BeoSub(mgr,descrName,tagName){}

void BeoSubLeakDetector::turnOpen(Angle, bool) {
        LERROR("Not Implemented!");
        return;
}

void BeoSubLeakDetector::advanceRel(float relPos, bool) {
        LERROR("Not Implemented!");
        return;
}

 void BeoSubLeakDetector::dropMarker(bool md) {
        LERROR("Not Implemented!");
        return;
}

 Image<PixRGB<byte> > BeoSubLeakDetector::grabImage(BeoSubCamera) const {
         LERROR("Not Implemented!");
         return Image< PixRGB<byte> >();

}

int main() {

        ModelManager manager("Testing Leak Detector");
        //create beochip
        nub::soft_ref<BeoChip> beochip(new BeoChip(manager)) ;
        manager.addSubComponent(beochip);

        //create beosub
        nub::soft_ref<BeoSubLeakDetector> sub(new BeoSubLeakDetector(manager,"BeoSubLeakDetector","BeoSubLeakDetector"));
        manager.addSubComponent(sub);

        //create Leak detector
        nub::soft_ref<LeakDetector> leakDetector(new LeakDetector(manager));
        manager.addSubComponent(leakDetector);

        //give leak detector beochip and sub reference
        leakDetector->setBeoChip(beochip);
        leakDetector->setBeoSub(sub);

        //create leak detector listener
        rutz::shared_ptr<LeakDetectorListener> ldLis(new LeakDetectorListener(leakDetector));
        rutz::shared_ptr<BeoChipListener> bcLis;
        bcLis.dynCastFrom(ldLis);
        beochip->setListener(bcLis);

        manager.start();

}


