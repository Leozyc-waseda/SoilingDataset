/*!@file AppDevices/test-EphysBoard.C receive data from EphysBoard and plot it continuously*/

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
// Primary maintainer for this file: Farhan Baluch <fbaluch@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-EphysBoard.C $
// $Id:  $
//

#include "Component/ModelManager.H"
#include "Devices/Serial.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Transport/FrameInfo.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameOstream.H"
#include "Util/Timer.H"
#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <math.h>
#include <fstream>
#include <queue>
#include <pthread.h>
#include <sys/soundcard.h>

std::queue<unsigned char*> dataCache;
std::vector<float> signalWin;
long int cnt = 0;
std::ofstream outFile("Ephysdata.dat");
pthread_mutex_t qmutex_data;
pthread_mutex_t mutexDone;
bool done=0;

/***** static void* getData(void *arg)  gets data fast over USB**/ 
// ######################################################################
static void* getData(void *)
{
  //int my_fd = (intptr_t)fd;
  
  int numBytes = 5;
  unsigned char frame[numBytes-1];
  char start[2];

  std::string USBDEV = "/dev/ttyACM0";
  
  int fd;
  int status;

  //setup a new file read manually 
  if(!(fd = open(USBDEV.c_str(), O_RDONLY)))
    perror("error opening device");  
  
  if(ioctl(fd, TIOCMGET, &status) == -1)
    LFATAL("TIOCMGET faileed: %s\n",strerror(errno));

  int fReturn;
  Timer tim;
  tim.reset();
  while(tim.getSecs() < 1)
    { 
      start[0] = '1';
       
      //wait for sync
      while (start[0] != '0')
        {
          fReturn = read(fd,start,1);
          //    printf("\nwaiting sync %s bytes returned %d",start,fReturn);
        }
      ///printf("synced");
       //now read bytes
      fReturn = read(fd,frame,numBytes-1); 
      if (fReturn == (numBytes-1))
        {
          pthread_mutex_lock(&qmutex_data);
          frame[numBytes-1] = '\0';
          dataCache.push(frame);
          // printf("read %d bytes start is %s",fReturn,start);
          pthread_mutex_unlock(&qmutex_data);
          printf("framepush: %s\n",frame);
        }       
      else
        {
          // printf("bad read");
        }
    }
  
  pthread_mutex_lock(&mutexDone);
  done = true;
  pthread_mutex_unlock(&mutexDone);
 return NULL;
}

// ######################################################################
//this process frame assumes frame starts with 0 and in block of 4 bytes
//msb1 msb2 lsb1 lsb2 
void processFrame(unsigned char *frame)
{
  int chunk = 4;
  unsigned char block[4];
  
  std::string sFrame = std::string(reinterpret_cast<const char*> (frame));

  printf("\n%s",sFrame.c_str());
  for (int i=0; i < (int)sFrame.length()-1; i+=4)
    {
      //LINFO("byte %d",i);
      for (int kk=0; kk<chunk; kk++)
        {
          unsigned char iFrame = sFrame[i+kk];
          if (kk <=1)
            iFrame = iFrame - 97;
          else
            iFrame = iFrame - 65;
          
          block[kk] = iFrame;
          //LINFO("byte %d: %s",i,resultStr);          
        }
      
      unsigned int data = (block[0] << 12 | block[1] << 8) | 
        (block[2] << 4 | block[3]);
      
      
      if (cnt > 1000) 
        {
          signalWin.erase(signalWin.begin());
          cnt = 0;
        }
      else
        cnt++;
      
      float dataF = data*(4.096F/65536);
      //LINFO("time %f, time/ms %f, val:%d,vol:%f",tim.getSecs(),
      //      tim.getMilliSecs(),data,dataF);
      outFile << dataF << std::endl; 
      signalWin.push_back(dataF);         
    }

}

// ######################################################################
//this process frame assumes frame starts with 0 and in block of 2 bytes 
void processFrame2(unsigned char *frame)
{
  

}
  

// ######################################################################
static int submain(int argc, char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test EphysBoard");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  Timer tim; 

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "Ephys board--", 0, 0) == false) 
    return(1);

  // let's get all our ModelComponent instances started:
  manager.start();
  usleep(1000);

  tim.reset();

  pthread_t acqThread;
  int rc;
  rc = pthread_create(&acqThread,NULL,&getData,NULL);
   if(rc)			/* could not create thread */
    {
        printf("\n ERROR: return code from pthread_create is %d \n", rc);
        exit(1);
    }

   printf("\n Created new thread (%d) ... \n", (int)acqThread);

   Image<PixRGB<byte> > itsPlot(700,500,ZEROS);
   
   bool imDone;
   pthread_mutex_lock(&mutexDone);
   imDone = done;
   pthread_mutex_unlock(&mutexDone);
   
   while(!imDone)
     {  
      
       unsigned char *frame=NULL;
               
       pthread_mutex_lock(&qmutex_data);
       if(dataCache.size())
         {
           //printf("here");
           frame = dataCache.front();
           dataCache.pop();
         }
       pthread_mutex_unlock(&qmutex_data);
                 
       if (frame != NULL)
         {
           
           processFrame(frame);
           itsPlot = linePlot(signalWin, 600,500,0.0F,0.0F, "Signal","voltage V",
                              "time/ms",PixRGB<byte>(0,0,0),
                              PixRGB<byte>(255,255,255),5,0);
         }
       else
         usleep(1000);
       
       
       //ofs->writeRGB(itsPlot,"Output",FrameInfo("output",SRC_POS));
        pthread_mutex_lock(&mutexDone);
        imDone = done;
        pthread_mutex_unlock(&mutexDone);
     }    
   pthread_exit(NULL);	
   
   // stop all our ModelComponents
   manager.stop();
   
   // all done!
   return 0;
   
}

extern "C" int main(const int argc, char** argv)
{
  try 
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }

  return 1;
}
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

