/*!@file RCBot/test-img.C
*/

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/test-img.C $
// $Id: test-img.C 15310 2012-06-01 02:29:24Z itti $
//

#include "Image/OpenCVUtil.H"
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ColorOps.H"
#include "Image/ShapeOps.H"
#include "Image/MathOps.H"
#include "Image/CutPaste.H"
#include "Image/Transforms.H"
#include "Media/FrameSeries.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include "Util/Timer.H"
#include "Learn/SOFM.H"
#include "GUI/XWinManaged.H"
#include "CMapDemo/SaliencyCMapMT.H"
#include "SIFT/ScaleSpace.H"
#include "SIFT/VisualObject.H"
#include "SIFT/Keypoint.H"
#include "SIFT/VisualObjectDB.H"
#include "Image/FourierEngine.H"
#include "RCBot/Motion/MotionEnergy.H"

#include <signal.h>


//! Number of frames over which average framerate is computed
#define NAVG 20

static bool goforever = true;
//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(0);}

ImageSet<float> bias(14);
ImageSet<float> cmap1(14);
ImageSet<float> cmap2(14);

struct        KeyPoint {
        int x;
        int y;
        float val;
};

#define ROI_SIZE 10
std::vector<KeyPoint>* getKeypoints(Image<float> &ima){
        Image<float> wima = ima; //Copy the image for debuging. Could be removed when actuly running because we dont
        //care about the image any more.
        std::vector<KeyPoint> *keypoints = new std::vector<KeyPoint>;

        //Try PCA

        //get the 10 most intersting keypoints and thier relationship
        for(int i=0; i<10; i++){

                 float val; Point2D<int> winner; findMax(wima, winner, val);
                 KeyPoint key;
                 key.x = winner.i;
                 key.y = winner.j;
                 key.val = val;
                 keypoints->push_back(key);

                 //IOR
                 drawDisk(wima, winner, ROI_SIZE, 0.0F);

        }


        return keypoints;

}


double compImg(std::vector< rutz::shared_ptr<Keypoint> > &obj1,
                std::vector< rutz::shared_ptr<Keypoint> > &obj2){


        if (obj1.size() != obj2.size()){
                LINFO("Objects size dont match %" ZU " %" ZU , obj1.size(), obj2.size());
                //return 999999;
        }

        std::vector<double> objM1;
        std::vector<double> objM2;
        std::vector<double> objDiff;


        //map cordinates
        if (obj1.size() > obj2.size()){
                for(unsigned int i=0; i<obj2.size(); i++){
                        objM1.push_back(obj1[i]->getO()*360+obj1[i]->getY()*160+obj1[i]->getX());
                        objM2.push_back(obj2[i]->getO()*360+obj2[i]->getY()*160+obj2[i]->getX());
                }
        } else {
                for(unsigned int i=0; i<obj1.size(); i++){
                        objM1.push_back(obj1[i]->getO()*360+obj1[i]->getY()*160+obj1[i]->getX());
                        objM2.push_back(obj2[i]->getO()*360+obj2[i]->getY()*160+obj2[i]->getX());
                }
        }


        //sort the arrays
        std::sort(objM1.begin(), objM1.end());
        std::sort(objM2.begin(), objM2.end());

        //find the diffrance
        for(unsigned int i=0; i<objM1.size(); i++){
                objDiff.push_back(fabs(objM1[i] - objM2[i]));
        }


        printf("M1: ");
        for(unsigned int i=0; i<objM1.size(); i++){
                printf("%f ", objM1[i]);
        }
        printf("\n");

        printf("M2: ");
        for(unsigned int i=0; i<objM2.size(); i++){
                printf("%f ", objM2[i]);
        }
        printf("\n");

        printf("Diff: ");
        for(unsigned int i=0; i<objDiff.size(); i++){
                printf("%f ", objDiff[i]);
        }
        printf("\n");

        //compute the relationship

        double sum = 0;
        printf("Delta: ");
        for(unsigned int i=0; i<objDiff.size()-1; i++){
                double diff = fabs(objDiff[i]-objDiff[i+1]);
                sum += diff;
                printf("%f ", diff);
        }
        printf("\n");


        LINFO("Diffrance %f", sum);

        return sum;
}



int main(int argc, char** argv)
{

        MYLOGVERB = LOG_INFO;  // suppress debug messages

        CORBA::ORB_ptr orb = CORBA::ORB_init(argc,argv,"omniORB4");

        // Instantiate a ModelManager:
        ModelManager manager("Test SOFM");

        // Instantiate our various ModelComponents:
        nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
        manager.addSubComponent(ifs);


        nub::ref<SaliencyMT> smt(new SaliencyMT(manager, orb, 0));
        manager.addSubComponent(smt);

        // Parse command-line:
        if (manager.parseCommandLine((const int)argc, (const char**)argv, "", 0, 0) == false)
                return(1);

        //catch signals and redirect them to terminate for clean exit:
        signal(SIGHUP, terminate); signal(SIGINT, terminate);
        signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
        signal(SIGALRM, terminate);


        // let's get all our ModelComponent instances started:
        manager.start();

        Timer masterclock;                // master clock for simulations


        masterclock.reset();
        ifs->update(masterclock.getSimTime());
        const Image< PixRGB<byte> > input = ifs->readRGB();
        const int w = input.getWidth();
        const int h = input.getHeight();

        LINFO("Input %ix%i", w, h);
        XWinManaged xwin2(Dims(256*3, 256), -1, -1, "SOFM Output");
        XWinManaged xwin(Dims(256*3, 256), -1, -1, "SOFM Output");

        // main loop:
        int ii=0;

        uint64 avgtime = 0; int avgn = 0; // for average framerate
        float fps = 0.0F;                 // to display framerate
        Timer tim;                        // for computation of framerate

        char info[1000];  // general text buffer for various info messages
        sprintf(info, "----");

        Image<PixRGB<byte> > disp(256*3, 256, NO_INIT);
        Image<PixRGB<byte> > disp2(256*3, 256, NO_INIT);


        //Get the image
        //Compute Saliency Map
        Image<PixRGB<byte> > cima = rescale(Raster::ReadRGB("/home/elazary/images/backyard/trial1/frame001000.ppm"), Dims(160,120));

        rutz::shared_ptr<VisualObject> voComp(new VisualObject("Loc3", "Loc3", cima));

        Image<float> SMapComp = smt->getSMap(cima);
        for(int i=0; i<14; i++){
                if (smt->cmaps[i].initialized()){
                        cmap1[i] = smt->cmaps[i];

                }
        }


        //inplaceNormalize(SMapComp, 0.0F, 255.0F);



        double minSMapDist = 999999999;
        int minSMap = 0;

        // double minSMapDistFFT = 99999999;
        // int minSMapFFT = 0;

        double minSiftDist = 99999999;
        int minSift = 0;

        while(goforever) {

                // read new image in?

                FrameState is = FRAME_NEXT;
                for(int i=0; i<100 && is != FRAME_COMPLETE; i++) //advance by  100 frames
                        is = ifs->update(masterclock.getSimTime());
                if (is == FRAME_COMPLETE)
                {
                        LINFO("quitting because input is complete");
                        break;
                }

                if (is == FRAME_NEXT || is == FRAME_FINAL) // new frame
                {
                        Image< PixRGB<byte> > ima = ifs->readRGB();

                        //Image<float> lum = luminance(ima);

                        //Compute Saliency Map
                        //


                        ///////////////////////////// SIFT ///////////////////////////

                        rutz::shared_ptr<VisualObject> vo(new VisualObject("Img", "Img", ima));

                        Image<PixRGB<byte> > img1 = cima;
                        Image<PixRGB<byte> > img2 = ima;


                        img1 = voComp->getKeypointImage();
                        img2 = vo->getKeypointImage();


                        std::vector< rutz::shared_ptr<Keypoint> > keypoints1 = voComp->getKeypoints();
                        std::vector< rutz::shared_ptr<Keypoint> > keypoints2 = vo->getKeypoints();

                        double imgDist = compImg(keypoints1, keypoints2);
                        //double imgDist = voComp->getFeatureDistSq(vo);

                        disp = rescale(ima, Dims(256, 256));
                        disp = concatX(disp, rescale(img1, Dims(256, 256)));
                        disp = concatX(disp, rescale(img2, Dims(256, 256)));



                        xwin.drawImage(disp);

                        getchar();


                        //////////////////////////////////////////////////////////////

                        /*
                        double distance = 0;
                        Image<float> SMap = smt->getSMap(ima);
                        for(int i=6; i<7; i++){
                                if (smt->cmaps[i].initialized()){
                                        cmap2[i] = smt->cmaps[i];

                                        inplaceNormalize(cmap1[i], 0.0F, 255.0F);
                                        inplaceNormalize(cmap2[i], 0.0F, 255.0F);

                                        std::vector<KeyPoint> *keypoints1;
                                        std::vector<KeyPoint> *keypoints2;

                                        keypoints1 = getKeypoints(cmap1[i]);
                                        keypoints2 = getKeypoints(cmap2[i]);

                                        distance += compImg(*keypoints1, *keypoints2);
                                        //Show the keypoints

                                        Image<PixRGB<byte> > SMapCompDisp = toRGB((Image<byte>)cmap2[i]);
                                        Image<PixRGB<byte> > SMapDisp = toRGB((Image<byte>)cmap1[i]);
                                        Point2D<int> LastLoc(0,0);

                                        for(unsigned int j=0; j<keypoints1->size(); j++){
                                        //        LINFO("Draw keypoint1 at (%i,%i) val=%f",
                                        //                        (*keypoints1)[j].x, (*keypoints1)[j].y, (*keypoints1)[j].val);
                                                drawDisk(SMapCompDisp, Point2D<int>((*keypoints1)[j].x,
                                                                        (*keypoints1)[j].y), ROI_SIZE, PixRGB<byte>(255,0,0));

                                                Point2D<int> NewLoc((*keypoints1)[j].x, (*keypoints1)[j].y);
                                                drawLine(SMapCompDisp, LastLoc, NewLoc, PixRGB<byte>(255,0,0));
                                                LastLoc = NewLoc;
                                        }
                                        LastLoc.i =0; LastLoc.j=0;
                                        for(unsigned int j=0; j<keypoints2->size(); j++){
                                        //        LINFO("Draw keypoint2 at (%i,%i) val=%f",
                                        //                        (*keypoints2)[j].x, (*keypoints2)[j].y, (*keypoints2)[j].val);
                                                drawDisk(SMapDisp, Point2D<int>((*keypoints2)[j].x,
                                                                        (*keypoints2)[j].y), ROI_SIZE, PixRGB<byte>(255,0,0));
                                                Point2D<int> NewLoc((*keypoints2)[j].x, (*keypoints2)[j].y);
                                                drawLine(SMapDisp, LastLoc, NewLoc, PixRGB<byte>(255,0,0));
                                                LastLoc = NewLoc;
                                        }

                                        disp = rescale(ima, Dims(256, 256));
                                        disp = concatX(disp, rescale(SMapCompDisp, Dims(256, 256)));
                                        disp = concatX(disp, rescale(SMapDisp, Dims(256, 256)));

                                        xwin.drawImage(disp);

                                        getchar();
                                }
                        }
                        */

                        //double SMapdist = distance(SMap, SMapComp);
      if (imgDist < minSMapDist){
        minSMapDist = imgDist;
        minSMap = ifs->frame();
      }

                        LINFO("Distance for %i = %f", ifs->frame(), imgDist);

                        // compute and show framerate and stats over the last NAVG frames:
                        avgtime += tim.getReset(); avgn ++;
                        if (avgn == NAVG)
                        {
                                fps = 1000.0F / float(avgtime) * float(avgn);
                                avgtime = 0; avgn = 0;
                        }

                        // create an info string:
                        sprintf(info, "%06u %.1ffps  ", ii++, fps);


                        ii++;
                        //          getchar();
                }


        }

        LINFO("BestSmapMatch %i SMapDist = %f", minSMap, minSMapDist);
        // LINFO("BestSmapFFTMatch %i SMapDist = %f", minSMapFFT, minSMapDistFFT);
        LINFO("BestSiftMatch %i SiftDist = %f", minSift, minSiftDist);

        // stop all our ModelComponents
        manager.stop();

        // all done!
        return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
