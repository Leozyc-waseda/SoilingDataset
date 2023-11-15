/*!@file SceneUnderstanding/SceneUnderstanding.C  */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/SceneUnderstanding.C $
// $Id: SceneUnderstanding.C 13413 2010-05-15 21:00:11Z itti $
//

#ifndef SCENEUNDERSTANDING_SCENEUNDERSTANDING_C_DEFINED
#define SCENEUNDERSTANDING_SCENEUNDERSTANDING_C_DEFINED

#include "plugins/SceneUnderstanding/SceneUnderstanding.H"

#include "Channels/SubmapAlgorithmBiased.H"
#include "ObjRec/BayesianBiaser.H"

#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

// ######################################################################
SceneUnderstanding::SceneUnderstanding(ModelManager *mgr, nub::ref<StdBrain> &brain) :
  itsMgr(mgr), itsBrain(brain)
{

  //generate a complexChannel from Visual cortex to pass to DescriptorVec
  ComplexChannel *cc = &*dynCastWeak<ComplexChannel>(itsBrain->getVC());

  OptionManager *mm = (OptionManager *)itsMgr;

  //get the seq component
  nub::ref<SimEventQueueConfigurator> seqc =
    dynCastWeak<SimEventQueueConfigurator>(itsMgr->subComponent("SimEventQueueConfigurator"));
  itsSEQ = seqc->getQ();

  LINFO("Start 1222");
  //Build the descriptor vector
  itsDescriptorVec = new DescriptorVec(*mm,
      "Descriptor Vector", "DecscriptorVec", cc);


  //set up featuers and 2 classes
  itsBayes = new Bayes(itsDescriptorVec->getFVSize(), 0); //start with no classes


  //get a new working memory
  itsWorkingMemory = new WorkingMemory;

  //set the feature names
  for (uint i = 0; i < cc->numSubmaps(); i++)
  {
    const std::string name = cc->getSubmapName(i);
    itsBayes->setFeatureName(i, name.c_str());
  }

  //The prolog engine
  itsProlog = new SWIProlog(0, NULL);


}

// ######################################################################
SceneUnderstanding::~SceneUnderstanding()
{

}

// ######################################################################
nub::ref<StdBrain> SceneUnderstanding::getBrainPtr()
{
  return itsBrain;
}

// ######################################################################
Bayes* SceneUnderstanding::getBayesPtr()
{
  return itsBayes;
}

// ######################################################################
SWIProlog* SceneUnderstanding::getPrologPtr()
{
  return itsProlog;
}

// ######################################################################
DescriptorVec* SceneUnderstanding::getDescriptorVecPtr()
{
  return itsDescriptorVec;
}

// ######################################################################
void SceneUnderstanding::setImage(Image<PixRGB<byte> > &img)
{
  itsImg = img;
  itsBrain->input(img, itsSEQ);
  itsWorkingMemory->setImage(img);
  itsDescriptorVec->setInputImg(img);
}

// ######################################################################
float SceneUnderstanding::evolveBrain()
{
  float interestLevel = 0.0F;

  //if (itsMgr->started() && img.initialized()){         //give the image to the brain
  if (itsImg.initialized()){         //give the image to the brain
    //itsBrain->time();

    LINFO("No new input");
    bool keep_going = true;
    while (keep_going){
      itsBrain->evolve(itsSEQ);
      const SimStatus status = itsSEQ->evolve();
      if (status == SIM_BREAK) {
        LINFO("V %d\n", (int)(itsSEQ->now().msecs()) );
        keep_going = false;
      }
      if (itsBrain->gotCovertShift()) // new attended location
      {

        const Point2D<int> winner = itsBrain->getLastCovertPos();
        const float winV = itsBrain->getLastCovertAgmV();

        interestLevel = (winV * 1000.0f) * 1; //the gain shold be based on context
        LINFO("##### Winner (%d,%d) at %fms : interestLevel=%f#####",
            winner.i, winner.j, itsSEQ->now().msecs(),interestLevel);


        keep_going = false;

        itsCurrentFoveaLoc = winner;
      }
      if (itsSEQ->now().secs() > 3.0) {
        LINFO("##### Time limit reached #####");
        keep_going = false;
      }
      LINFO("Evolve brain");
    }

  }

  return interestLevel;
}

// ######################################################################
Point2D<int> SceneUnderstanding::getFoveaLoc()
{
  return itsCurrentFoveaLoc;
}

// ######################################################################
std::vector<double> SceneUnderstanding::getFV(Point2D<int> foveaLoc)
{

  //build descriptive vector
  itsDescriptorVec->setFovea(foveaLoc);
  Dims foveaSize = itsDescriptorVec->getFoveaSize();

  itsDescriptorVec->buildRawDV();

  //get the resulting feature vector
  std::vector<double> FV = itsDescriptorVec->getFV();

  return FV;

}

// ######################################################################
int SceneUnderstanding::classifyFovea(Point2D<int> foveaLoc, double *prob)
{

  std::vector<double> FV = getFV(foveaLoc);

  int cls = itsBayes->classify(FV, prob);

  return cls;

}

// ######################################################################
void SceneUnderstanding::learn(Point2D<int> foveaLoc, const char *name)
{

  std::vector<double> FV = getFV(foveaLoc);
  itsBayes->learn(FV, name);

}

// ######################################################################
bool SceneUnderstanding::biasFor(const char *name)
{
  int cls = itsBayes->getClassId(name);
  if (cls != -1 ) //we know about this object
  {

    ComplexChannel *cc = &*dynCastWeak<ComplexChannel>(itsBrain->getVC());

    //Set mean and sigma to bias submap
    BayesianBiaser bb(*itsBayes, cls, -1, true);
    cc->accept(bb);

    //Tell the channels we want to bias
    setSubmapAlgorithmBiased(*cc);

    itsBrain->input(itsImg, itsSEQ); //Set a new input to the brain
    itsDescriptorVec->setInputImg(itsImg);
    return true;
  } else {
    LINFO("Object %s is unknown", name);
    return false;
  }

}

// ######################################################################
std::string SceneUnderstanding::highOrderRec()
{

  static bool moreQuestions = true;
  std::vector<std::string> args;
  std::string question;
  std::string sceneType;

  if(moreQuestions)
  {

    //get the scene type
    args.clear();
    args.push_back(std::string());
    if( itsProlog->query("go", args) )
    {
      LINFO("Scene is %s", args[0].c_str());
      sceneType = args[0];
    } else {
      LINFO("Something is worng");
    }

    //check if we have any questions
    args.clear();
    args.push_back(std::string());
    if( itsProlog->query("getQuestion", args) )
    {
      LINFO("Question is %s", args[0].c_str());
      question = args[0];
    } else {
      LINFO("No more Question" );
      moreQuestions = false;
    }

    if (moreQuestions)
    {
      //get the command and object to look for
      std::string cmd = question.substr(0,2);
      std::string objLookingFor = question.substr(2,question.size());

      sceneType += std::string(".   Looking for: ") + question;

      switch(cmd[0])
      {
        case 'e':   //check if object exists
          //Attempt to answer the questions by finding the features
          if (biasFor(objLookingFor.c_str()) )
          {
            //for now only once time and give up
            evolveBrain(); //TODO: get intrest level and decide when to stop searching

            //recognize the object under the fovea
            int cls = classifyFovea(itsCurrentFoveaLoc);
            if (cls != -1)
            {
              std::string objName = std::string(itsBayes->getClassName(cls));
              std::string predicate;

              LINFO("Found object %s", objName.c_str());
              if (objName ==  objLookingFor)
              {
                predicate = "setYes"; //we found it
                sceneType += std::string(".   object found. ");

                PixRGB<byte> col(0,255,0);
                itsBrain->getSV()->setColorNormal(col);
              }
              else
              {
                predicate = "setNo";  //we did not find it
                sceneType += std::string(".   object not found. ");
                PixRGB<byte> col(255,255,0);
                itsBrain->getSV()->setColorNormal(col);
              }

              //Set what we know about the scene
              LINFO("Asserting %s %s", predicate.c_str(), objLookingFor.c_str());
              args.clear();
              std::string obj = "e_" + objLookingFor;
              args.push_back(obj);
              if( itsProlog->query((char *)predicate.c_str(), args) )
              {
                LINFO("Assert OK");
              } else {
                LINFO("Fatal Assertion");
              }
            }

          } else {
            LINFO("I am sorry, but I dont know anything about ");
          }
          break;

        case 'c': //count the number of times the object exists
          //Attempt to answer the questions by finding the features
          if (biasFor(objLookingFor.c_str()) )
          {
            //Set the obj count to 0 (if obj is not set) to indicate that we already looked for the object
            args.clear();
            std::string obj = "c_" + objLookingFor;
            args.push_back(obj);
            if( itsProlog->query("resetCount", args) )
            {
              LINFO("incCount OK");
            } else {
              LINFO("incCount");
            }

            float interestLevel = 10;
            int count = 0;
            for(int i=0; i<20 && interestLevel > 5.0F; i++)
            {
              interestLevel = evolveBrain();

              double prob=0;
              //recognize the object under the fovea
              int cls = classifyFovea(itsCurrentFoveaLoc, &prob);
              if (cls != -1)
              {
                std::string objName = std::string(itsBayes->getClassName(cls));
                std::string predicate;

                LINFO("%i: Found object %s(%f). Interest %f", i, objName.c_str(), prob, interestLevel);

                //remeber the object under the fovea
                itsWorkingMemory->setMemory(itsCurrentFoveaLoc, cls, itsDescriptorVec->getFoveaSize().h()/2);

                //Set what we know about the scene
                //TODO. need to optemize, since during our search we incounter objects
                //but we dont count for them. What happens to open that have we don't know about?
                //that is have very low probability
              //  if (objName ==  objLookingFor)
               // {
                  count++;
                  args.clear();
                  std::string obj = "c_" + objName; //objLookingFor;
                  args.push_back(obj);
                  if( itsProlog->query("incCount", args) )
                  {
                    LINFO("incCount OK");
                  } else {
                    LINFO("incCount");
                  }

                  //Set the FOA to a color indicating that we found the object
                  int confCol = int(-1.0*prob);

                  if (confCol > 255) confCol = 255;
                  confCol = 255-confCol;

                  PixRGB<byte> col(0,confCol,0);
                  itsBrain->getSV()->setColorNormal(col);
              //  }
              //  else
              //  {
              //    //Set the FOA to a color indicating that we did not find the object
              //    PixRGB<byte> col(255,255,0);
              //    itsBrain->getSV()->setColorNormal(col);

              //  }

              }
            }

            char num[100];
            sprintf(num, "%i", count);
            sceneType += std::string(".   found: ") + std::string(num);
          } else {
            LINFO("I am sorry, but I dont know anything about that");
          }
          break;
      }

      //show debug
      args.clear();
      if(itsProlog->query("debug", args) )
      {
        LINFO("Debug Ok");
      } else {
        LINFO("Fatal Question");
      }

      //empty the questions list
      args.clear();
      if(!itsProlog->query("undoQuestions", args) )
      {
        LINFO("Question Deleted");
      } else {
        LINFO("Fatal Question");
      }

    } else {
      //get the scene type
      args.clear();
      args.push_back(std::string());
      if( itsProlog->query("hypothesize", args) )
      {
        LINFO("Scene is %s", args[0].c_str());
        sceneType = args[0];
      } else {
        LINFO("Something is worng");
      }

      //No more questions, retract answers
      args.clear();
      if( itsProlog->query("undo", args) )
      {
        LINFO("Removed all Assertion");
      } else {
        LINFO("Can not remove all Assertion");
      }
      moreQuestions = true;

    }


  }

 // //restore FOA color back to normal
 // PixRGB<byte> col(255,255,0);
 // itsBrain->getSV()->setColorNormal(col);

  return sceneType;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //SCENEUNDERSTANDING_SCENEUNDERSTANDING_C_DEFINED

