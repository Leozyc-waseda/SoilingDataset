/*! @file Beobot/test-gistNav-master.C -- An autonomous visual navigation
  using image intensity amplitude spectrum components FFT (fast and free
  from fftw.com) via object FFTWWrapper. Computes outputs with a back-prop
  network (FFN.H&.C) and online learning.
  -- Christopher Ackerman, 7/30/2003                                    */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/test-gistNav-master.C $
// $Id: test-gistNav-master.C 15310 2012-06-01 02:29:24Z itti $

#include "Beowulf/Beowulf.H"
#include "Beowulf/BeowulfOpts.H"
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/KeyBoard.H"
#include "Devices/RadioDecoder.H"
#include "Devices/lcd.H"
#include "Gist/FFN.H"
#include "Image/FFTWWrapper.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/log.H"

#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#define OUTPATH "/home/tmp/7/gistNav/"
  // Current directory to write to for this grab

char dir[10];

// tell writer thread when to write a new image:
bool doWrite;

// tell grabber it is okay to grab
bool grabOk;

// tell writer thread to quit:
bool doQuit;

// tell grabbing loop that the user has pushed quit
bool quitGrab;

// current frame number:
int frame=0;

//!< Will turn false on interrupt signal
static bool goforever = true;

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s)
{  LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

#define HBLOCKS 4
#define WBLOCKS 4
#define DIMSINALL 640
#define DIMSIN 40
#define DIMSHID 90
#define DIMS1HID 40
#define DIMS2HID 40
#define DIMSOUT 13
#define IMAGEHEIGHT 120
#define IMAGEWIDTH 160
#define IMAGESIZE IMAGEHEIGHT*IMAGEWIDTH
#define WINDOWHEIGHT 30
#define WINDOWWIDTH 40
#define WINDOWSIZE WINDOWHEIGHT*WINDOWWIDTH
#define LEARNRATE 0.5
#define MOMENTUM 0.0

double compsteer, compspeed;
Image< PixRGB<byte> > ima(IMAGEWIDTH,IMAGEHEIGHT, NO_INIT);

// ######################################################################
void gzWriteFrame(int frameNum, Image< PixRGB<byte> >& imag)
{
  char temp[256];
  char head[256];

  head[0] = '\0';
  temp[0] = '\0';

  //  gzFile gzOut;
  FILE *fp;

  //To copy the RC data
  float taccel, tsteer;
  taccel = compspeed;
  tsteer = compsteer;

  // sprintf(temp, "./frame%06d.ppm.gz", frameNum);
  sprintf(temp, "./frame%06d.ppm", frameNum);
  LDEBUG("Filename: %s", temp);

  // Open the file using zlib gzopen routine
  // gzOut = gzopen (temp, "wb");
  fp = fopen(temp, "wb");

  // if( gzOut == NULL )    //Check status of file open operation
  if( fp == NULL )    //Check status of file open operation
    LFATAL("There was an error opening the gzip file for writing");

  // Write out the header information
  sprintf(temp, "P6\n");

  strcpy(head, temp);

  sprintf(temp, "#ACCEL:%-10f STEER:%-10f\n", taccel, tsteer);
  strcat(head, temp);
  sprintf(temp, "%d %d\n255\n", imag.getWidth(), imag.getHeight() );
  strcat(head, temp);

  // if( !gzwrite(gzOut, head, strlen(head)) )
  if( !fwrite(head, 1, strlen(head), fp) )
    LFATAL("There was an error gzwriting %" ZU " bytes", strlen(head));

  if( !fwrite(reinterpret_cast<void*>(imag.getArrayPtr()), 1,
              3*imag.getSize(), fp))
    // if( !gzwrite(gzOut, reinterpret_cast<void*>(imag.getArrayPtr()),
    //               3*imag.getSize()))
    LFATAL("There was an error gzwriting %d bytes", 3*imag.getSize());

  //  gzclose(gzOut);
  fclose(fp);
}

// ######################################################################
void* imageWriterThread(void *junk)
{
  ////  while(doQuit == false){
  if (doWrite)
    {
      Image< PixRGB<byte> > currFrame = ima;
      // for a deep copy of the image data:
      Image< PixRGB<byte> >::iterator data = currFrame.beginw();
      data ++; // keep g++ happy
      grabOk = true;

      gzWriteFrame(frame++, currFrame);

      doWrite = false;
    }
  else
    usleep(500);
  ////    }

  // quit the thread:
  ////  pthread_exit(0);
  return NULL;
}

// ######################################################################
void grabCycleInit( KeyBoard& myKB, nub::soft_ref<lcd> lcdDisp,
                    nub::soft_ref<FrameIstream>& gb )
{
  char tmp[100];
  int i = 0;

  //We want to create a new directory to put frames in
  //But make sure that we don't overwrite any directories
  //That already exist

  if( chdir( OUTPATH ) ) //If there's a problem cding to ouput dir
    {
      LINFO("There was an error changing to directory: %s", OUTPATH);
      lcdDisp->clear();
      lcdDisp->printf(0, 0, " Output directory" );
      lcdDisp->printf(0, 1, " Does not exist!" );
      lcdDisp->printf(0, 2, " Trying to create. ");
      sleep(1);

      if( mkdir( OUTPATH, S_IRUSR | S_IWUSR ) != 0 )
        {
          LINFO("There was an error creating directory: %s", OUTPATH);
          lcdDisp->clear();
          lcdDisp->printf(0, 0, " mkdir() failed! ");
          lcdDisp->printf(0, 1, " bailing out... ");
          sleep(1);
          return;
        }
      if (chdir( OUTPATH )) LFATAL("chdir error");
    }

  i = 0;
  do
    {
      sprintf( tmp, "%d", i++ );
    }
  while ( mkdir(tmp, S_IRUSR | S_IWUSR | S_IEXEC ) != 0 );
  sprintf( tmp, "%s%d", OUTPATH, i-1 );
  if (chdir( tmp )) LFATAL("chdir error");

  char * ptr = getcwd( NULL, 0 );
  LDEBUG( "Working directory is %s", ptr);
  free( ptr );

  frame = 0; doQuit = false;
  doWrite = false; grabOk = true; quitGrab = false;  // start writer thread:

  ////  pthread_t runner;
  ////  pthread_create(&runner, NULL, &imageWriterThread, NULL);
}

// ######################################################################
void grabCycle( KeyBoard& myKB, nub::soft_ref<lcd> lcdDisp,
                nub::soft_ref<FrameIstream>& gb )
{
  // wait for writer thread to be ready and tell it to write frame:
  while (doWrite) usleep(500);
  grabOk = false;  // not okay to grab until writer thread authorizes us
  doWrite = true;  // trigger write of current frame
  imageWriterThread(NULL);
  while(grabOk == false) usleep(500);
}

// ######################################################################
int main(const int argc, const char **argv)
{
  int i, j, h, l;
  bool updateweights = false;
  double steer, speed;
  double rcspeed, rcsteer;

  // Choose steering probabilisticly,
  // with the probabilities coming from the relative strengths
  // of the network outputs
  srand((unsigned)time(0));

  int flag=0;
  long ctr=0;
  double total,mean,std;
  //double orient,open,expand,obstacle,avgsteer=0.0,laststeer=0.0;
  //double safety=0.5;

  /**************** 0: Setup *********************/
  Image<double> trueout(1, DIMSOUT, ZEROS);
  double masks[DIMSIN][WINDOWSIZE];
  Image<double> meanarr(1, DIMSIN, ZEROS); // normalizer
  Image<double> stdarr(1, DIMSIN,ZEROS);  // normalizer
  //double minarr[DIMSIN];// normalizer
  //double maxarr[DIMSIN];// normalizer
  //double range[DIMSIN]; // normalizer
  double evecs[DIMSIN][DIMSINALL];
  Image<double> allfeatures(1, DIMSINALL, ZEROS);
  Image<double> features(1, DIMSIN, ZEROS);// amp-dims input

  double **magspec = (double **)malloc((WINDOWHEIGHT)*sizeof(double *));
  if(magspec == NULL)
    LINFO("Error allocating memory for magspec");
  for(i = 0; i < WINDOWHEIGHT; i++)
  {
    magspec[i] = (double *)malloc((WINDOWWIDTH/2+1)*sizeof(double));
    if(magspec[i] == NULL)
      LINFO("Error allocating memory for magspec[i]");
  }

  double *bwimage = (double *)malloc(IMAGESIZE*sizeof(double));
  if(bwimage == NULL)
    LINFO("Error allocating memory for bwimage");
  else
    LINFO("okay");

  double *bwimage_sm = (double *)malloc(WINDOWSIZE*sizeof(double));

  FeedForwardNetwork *ffn_dm = new FeedForwardNetwork();

  // I was going to do a command line thing
  // where you could tell it to initialize with random weights in a range
  // instead of using a weight file
  /*(if (argc==2)
    if(strcmp(argv[1],"-i")==0)
    ffn_dm->init(atof(argv[1]),DIMSIN,DIMSHID,DIMSOUT,LEARNRATE);
    else{
    LERROR("Invalid switch");
    exit(1);
    }
    else*/

  // But when I tried the manager business choked on the new parameters.
  // So now it will initialize randomly within a hard-coded range
  // if it doesn't find the appropriately named weight files
  // if I am using a weight file computed offline in Matlab,
  // I pass it in a .mat file.
  // I also have to make some changes (uncomment some code)
  // in FFN.C to handle to Matlab file format. That kind of sucks.
  // If it's too much of a pain, then just convert the Matlab file to the
  // appropriate format and name in .dat.
  //ffn_dm->init("/lab/bb/data/nnwhdm-T.mat","/lab/bb/data/nnwodm-T.mat",
  //DIMSIN,DIMSHID,DIMSOUT,LEARNRATE,MOMENTUM);
  //ffn_dm->init("/lab/bb/data/nnwhdm.dat","/lab/bb/data/nnwodm.dat",
  //DIMSIN,DIMSHID,DIMSOUT,LEARNRATE,MOMENTUM);
  //ffn_dm->init3L("/lab/bb/data/nnwh1dm-T.mat","/lab/bb/data/nnwh2dm-T.mat",
  //"/lab/bb/data/nnwodm-T.mat",DIMSIN,DIMS1HID,DIMS2HID,DIMSOUT,LEARNRATE);
  ffn_dm->init3L("/lab/bb/data/nnwh1dm.dat","/lab/bb/data/nnwh2dm.dat",
    "/lab/bb/data/nnwodm.dat",DIMSIN,DIMS1HID,DIMS2HID,DIMSOUT,LEARNRATE,0.0);

  FFTWWrapper *fftw = new FFTWWrapper( WINDOWWIDTH,WINDOWHEIGHT);
  fftw->init(bwimage_sm);

  FILE *fp=fopen("/lab/bb/data/meanarrall.mat","rb");
  if(fp==NULL) LFATAL("Mean file not found");

  fseek(fp, 0, SEEK_END);
  long pos = ftell(fp);
  fseek(fp,pos-(DIMSIN*sizeof(double)), SEEK_SET);
  // FIX below
  //for(i = 0; i < DIMSIN; i++) fread(&meanarr[i],sizeof(double),1,fp);
  fclose(fp);

  // These four are Matlab-created files.
  // Note the way to find the beginning of the data
  fp = fopen("/lab/bb/data/stdarrall.mat","rb");
  if(fp == NULL) LFATAL("STD file not found");
  fseek(fp, 0, SEEK_END);
  pos=ftell(fp);
  fseek(fp,pos-(DIMSIN*sizeof(double)), SEEK_SET);
  // FIX below
  //  for(i = 0; i < DIMSIN; i++) fread(&stdarr[i],sizeof(double),1,fp);
  fclose(fp);

  //for(i = 0; i < DIMSIN; i++)
  //        range[i] = maxarr[i]-minarr[i];

  fp = fopen("/lab/bb/data/G2small.mat","rb");
  //G = fftshift(reshape(H:,i)...G2=reshape(G(:,:,i)'
  //...must have super low values from matlab trimmed
  if(fp == NULL) LFATAL("Mask file not found");
  fseek(fp, 0, SEEK_END);
  pos = ftell(fp);
  fseek(fp,pos-(DIMSIN*WINDOWSIZE*sizeof(double)), SEEK_SET);
  for(i = 0; i < DIMSIN; i++)
    if (fread(masks[i],sizeof(double),WINDOWSIZE,fp) != WINDOWSIZE) LFATAL("fread error");
  fclose(fp);

  fp=fopen("/lab/bb/data/evec40all.mat","rb");
  if(fp==NULL) LFATAL("Eigenvector file not found");
  fseek(fp, 0, SEEK_END);
  pos=ftell(fp);
  fseek(fp,pos-(DIMSIN*DIMSINALL*sizeof(double)), SEEK_SET);
  for(i=0;i<DIMSIN;i++)
    if (fread(evecs[i],sizeof(double),DIMSINALL,fp) != DIMSINALL) LFATAL("fread error");
  fclose(fp);

  double *iptr = NULL;

  TCPmessage smsg;      // buffer to send messages to nodes;
  // in my case speed and steering...
  // when the ethernet was working, counter and status stuff for lcd display now

  KeyBoard myKB;
  LINFO("Press A to do initialization"); // this is to get the beobot outside
    // before you start camera so it adjust to luminance properly
  while (myKB.getKey( true ) != KBD_KEY1) // actually that's no longer true.
    // Mostly it's to make sure you're ready.
    // As it happens, the camera adjusts itself well with time anyway
  LINFO("\a"); // beep

  // instantiate a model manager:
  ModelManager manager("Gist Navigator - Master");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  //uncomment when ethernet is fixed --changed bb1bg to bb1ag
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
  manager.addSubComponent(beo);
  manager.setOptionValString(&OPT_BeowulfSlaveNames, "bb1ag");

  nub::soft_ref<RadioDecoder> rd(new RadioDecoder(manager));
  manager.addSubComponent(rd);

  nub::soft_ref<lcd> lcdDisp(new lcd(manager));
  lcdDisp->setModelParamString("SerialPortDevName",
                               "/dev/ttyS0", MC_RECURSE);
  manager.addSubComponent(lcdDisp);

  // choose an IEEE1394grabber by default, and a few custom grabbing
  // defaults, for backward compatibility with an older version of
  // this program:
  manager.setOptionValString(&OPT_FrameGrabberType, "1394");
  manager.setOptionValString(&OPT_FrameGrabberDims, "160x120");
  manager.setOptionValString(&OPT_FrameGrabberMode, "YUV444");
  manager.setOptionValString(&OPT_FrameGrabberNbuf, "20");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("No grabber. Why does god hate me?");
  LINFO("GB width = %d height = %d\n",gb->getWidth(),gb->getHeight());

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP,  terminate);  signal(SIGINT,  terminate);
  signal(SIGQUIT, terminate);  signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // let's get all our ModelComponent instances started:
  manager.start();

  // Calibrate the controls' 0 position
  lcdDisp->clear();
  lcdDisp->printf(0, 1, "Calibrate Controls");
  lcdDisp->printf(0, 2, "Rest Controls NOW!");
  rd->zeroCalibrate();

  // Calibrate range of control swing
  lcdDisp->clear();
  lcdDisp->printf(0, 1, "Calibrate Controls");
  lcdDisp->printf(0, 2, "Full Swing NOW!");
  rd->rangeCalibrate();

  // Save our configuration data so we don't have to calibrate again (Ha!)
  manager.saveConfig();

  lcdDisp->clear();
  lcdDisp->printf(0, 2, "Press 2 (B) to Start");
  smsg.reset(ctr, 0);
  smsg.addInt32(0);
  beo->send(0, smsg);
  while (myKB.getKey( true ) != KBD_KEY2)
    printf("\a"); // beep
  lcdDisp->clear();
  printf("\a\a\a"); // beep

  grabCycleInit( myKB, lcdDisp, gb );
  /*******************************************************/

  Timer tim(1000000);
  tim.reset();
  uint64 end, start=tim.get();
  uint64 endpause, startpause=0, totalpause=0;
  double elapsed;
  double lastspeed = 0.4; // this is the default speed it starts out with.
  double totspeed = 0.0;
  unsigned long speedctr = 0;
  bool setspeed = false;
  int keypress = KBD_NONE;
  bool paused = false;
  while(!flag && goforever)
  {
    // this is to allow me to stop and change batteries,
    // etc w/o shutting down or causing it to learn to stop
    if( myKB.getKey( false ) == KBD_KEY3 || keypress == KBD_KEY3)
    {
      printf("pause\n");
      keypress=KBD_NONE;
      usleep(500);//to avoid double hits
      if(paused)
      {
        paused=false;
        endpause=tim.get();
        totalpause+=endpause-startpause;
      }
      else
      {
        smsg.reset(ctr, 0);  //when ethernet is fixed,
        //remove these 3 line and update lcd here
        smsg.addInt32(-1);
        beo->send(0, smsg);
        paused=true;
        startpause=tim.get();
      }
    }
    if(paused) continue;

    /****************** 1: Grab Frame ******************/
    ima = gb->readRGB();
    Image< PixRGB<byte> >::const_iterator data = ima.begin();//w();
    /***************************************************/

    /************** 2: Do Image Preproc ****************/
    // extract Value (brightness/luminance/intensity/whatever) from HSV
    total = 0.0;
    iptr  = bwimage;
    while (data != ima.end())
    {
      double max = 0.0;
      if (data->red()   > max) max = data->red();
      if (data->green() > max) max = data->green();
      if (data->blue()  > max) max = data->blue();
      *iptr = max;
      data++;
      total += *iptr++;
    }

    //normalize
    mean = total / ((double)IMAGESIZE);
    std = 0.0;
    for(i = 0; i < IMAGESIZE; i++)
      std += pow(bwimage[i]-mean,2);
    std /= IMAGESIZE-1;
    std = sqrt(std);
    for(i = 0; i < IMAGESIZE; i++)
    {
      bwimage[i] -= mean;
      bwimage[i] /= std;
    }

    /****************************************************/

    /***** 3: Do FFT and convolution for each block *****/
    for(h = 0; h < HBLOCKS; h++)
    {
      for(l = 0; l < WBLOCKS; l++)
      {
        //FFT
        iptr = bwimage_sm;
        for(i = 0; i < WINDOWHEIGHT; i++)
          for(j = 0; j < WINDOWWIDTH; j++)
            *iptr++ = bwimage[h*WINDOWHEIGHT*IMAGEWIDTH+
                              l*WINDOWWIDTH + i*IMAGEWIDTH + j];
        fftw->compute(magspec);

        // Convolve with masks
        // This looks a bit more complicated than you might think
        // because I'm saving some time by not multiplying by the
        // whole image. The masks only cover a bit over half the image.
        for(i=0;i<DIMSIN;i++)
        {
          const int halfwidth=WINDOWWIDTH/2+1;
          const int halfheight=WINDOWHEIGHT/2;

          int heighttop=halfheight;
          switch (i%8)
          {
          case 0:
            heighttop = halfheight;
            break;
          case 1:
          case 7:
            heighttop = (int)((double)halfheight*0.75);
            break;
          case 2:
          case 6:
            heighttop = (int)((double)halfheight*0.5);
            break;
          case 3:
          case 5:
            heighttop = (int)((double)halfheight*0.25);
            break;
          case 4:
            heighttop = 0;
            break;
          }

          //some portion of top half
          for(int j = 0; j < heighttop; j++)
          {//left side
            for(int k = 0; k < halfwidth; k++)
            {
              int index =  h*WBLOCKS*DIMSIN+l*DIMSIN+i;
              allfeatures.setVal
                (0, index, allfeatures.getVal(0, index) +
                 magspec[j][k]*masks[i][j*WINDOWWIDTH+k]);
            }
          }
          if(heighttop>0)
            for(int k = halfwidth; k < WINDOWWIDTH; k++)//top row
              {
                int index = h*WBLOCKS*DIMSIN+l*DIMSIN+i;
                allfeatures.setVal
                  (0, index, allfeatures.getVal(0,index) +
                   magspec[0][WINDOWWIDTH-k]*masks[i][k]);
              }

          for(int j = 1; j < heighttop; j++)
          {//right side
            for(int k = halfwidth; k < WINDOWWIDTH; k++)
            {
              int index = h*WBLOCKS*DIMSIN+l*DIMSIN+i;
              allfeatures.setVal
                (0, index, allfeatures.getVal(0,index)  +
                 magspec[WINDOWHEIGHT-j][WINDOWWIDTH-k]*
                 masks[i][j*WINDOWWIDTH+k] );
            }
          }

          //bottom half
          for(int j = halfheight; j < WINDOWHEIGHT; j++)
          {//left side
            for(int k = 0; k < halfwidth; k++)
            {
              int index = h*WBLOCKS*DIMSIN+l*DIMSIN+i;
              allfeatures.setVal
                (0, index, allfeatures.getVal(0,index) +
                 magspec[j][k]*masks[i][j*WINDOWWIDTH+k]);
            }
          }

          for(int j = halfheight; j < WINDOWHEIGHT; j++)
          {//right side
            for(int k = halfwidth; k < WINDOWWIDTH; k++)
            {
              int index = h*WBLOCKS*DIMSIN+l*DIMSIN+i;
              allfeatures.setVal
                (0, index, allfeatures.getVal(0, index) +
                 magspec[halfheight-(j-halfheight)][WINDOWWIDTH-k]*
                 masks[i][j*WINDOWWIDTH+k]);
            }
          }
        }
      }
    }
    /************************************************/

    /********** 4: Project onto eigenvectors ********/
    for(i = 0; i < DIMSIN; i++)
      for(j = 0; j < DIMSINALL; j++)
        features.setVal
          (0, i, features.getVal(0,i) + evecs[i][j]*allfeatures.getVal(0,j));
    /************************************************/

    /****************** 5: FFN for Dims *************/
    features = (features - meanarr)/stdarr;
    ffn_dm->run3L(features);
    /************************************************/

    /************ 6: Compute Motor Commands *********/
    ////        speed=ffn_dm->out[DIMSOUT-1]*2.0-1.0;
    // eh, make when I was computing speed to, rather than hacking it
    /* speed hack: if I'm giving it a speed command with the remote,
       it goes with that, otherwise it uses it's last speed value
      which is either the default (if I haven't given any speed commands yet)
      or the average of all the speed commands I gave last time*/
    rcspeed = rd->getVal(0);
    if(fabs(rcspeed) > 0.1)
    {
      setspeed = true;
      lastspeed = rcspeed;
      totspeed += rcspeed;
      speedctr++;
    }
    else
    {
      if(setspeed)
      {
        setspeed = false;
        lastspeed = totspeed/(double)speedctr;
        totspeed = 0.0;
        speedctr = 0;
      }
    }
    speed = lastspeed;

    /*  // I could just choose the average output, but...
        double mysum=0.0;steer=0.0;
        for(i=0;i<DIMSOUT-1;i++){
        mysum+=ffn_dm->out[i];
        steer+=((double)(i+1))*ffn_dm->out[i];
        }
        steer=steer/mysum;
        steer=((steer-1.0)/((double)DIMSOUT-1.0[2.0])*2.0-1.0)*-1.0;
    */

    // I choose the output probabilisticly
    double target=(double)rand()/(double)RAND_MAX;
    double mysum=0.0;double tprob=0.0;
    for(i=0;i<DIMSOUT-1;i++)
      mysum+=ffn_dm->getOutput().getVal(0,i);
    for(i=0;i<DIMSOUT-1;i++)
    {
      tprob+=ffn_dm->getOutput().getVal(0,i)/mysum;
      if (tprob>target)
        break;
    }
    steer=((((double)i+1.0)-1.0)/((double)DIMSOUT-1.0/*2.0*/)*2.0-1.0)*-1.0;
    //the commented 2 is left over from when speed was the last output

    /***********************************************/

    /************ 7: Execute Motor Control *********/
    compspeed = speed; //this is what gets written to the image file
    compsteer = steer;
    printf("computed speed=%f, steer=%f ctr=%ld\n",speed,steer,ctr+1);

    updateweights=false;
    rcsteer=rd->getVal(1);
    if(fabs(rcspeed)>0.1 || fabs(rcsteer)>0.1){
      printf("big rc signal\n");
      updateweights=true;
      /*if(fabs(rcspeed)>0.1)*///speed=rcspeed;
      /*if(fabs(rcsteer)>0.1)*/steer=rcsteer;
    }
    printf("rc speed=%f, steer=%f ctr=%ld\n",rcspeed,rcsteer,ctr+1);

    // call servo controller
    smsg.reset(ctr, 0);
    ////      steer=steer-steer*fabs(steer)/5;
    smsg.addDouble(steer);
    //        smsg.addDouble((speed*safety));
    //        speed=speed-speed*fabs(speed)/1.6666666667;//2;//max of 4
    smsg.addDouble(speed-speed*fabs(speed)/1.6666666667);
    beo->send(0, smsg);  // send off to slave

    // backprop
    if(updateweights==true){
      trueout.clear(0.0);
      int on;
      on=(int)round(((steer*-1.0+1.0)/2.0)*((double)DIMSOUT-1.0/*2.0*/)+1.0);
      trueout.setVal(0, on-1, 1.0);
      if (on>1)
        trueout.setVal(0, on-2, 0.5);//give some weight to nearby orientations.
        // My thinking is this suppies a little bit of ordering to a system
      if (on>2)
        trueout.setVal(0, on-3, 0.25);//assuming choosing probabilisticly
        // rather than averaging) that regards eg hard right as equally distant
      if (on<DIMSOUT)
        trueout.setVal(0, on, 0.5);//from slightly less hard right as from left
      if (on<DIMSOUT-1)
        trueout.setVal(0, on+1, 0.25);
      /////             trueout[DIMSOUT-1]=(speed+1.0)/2.0;
      ffn_dm->backprop3L(trueout);
    }

    // administrative stuff
    if(ctr%90==0 && ctr!=0)grabCycle( myKB, lcdDisp, gb );
    // I save weights occasionally to prevent
    // the obligatory battery deaths from becoming catastrophic
    if(ctr%5000==0 && ctr!=0)
      ffn_dm->write3L("/lab/bb/data/nnwh1dm.dat",
                      "/lab/bb/data/nnwh2dm.dat", "/lab/bb/data/nnwodm.dat");
      // so I don't lose my work when it crashes

    keypress=myKB.getKey( false );
    if( keypress == KBD_KEY4 ){
      flag=1;
      doQuit=true;
      LDEBUG("quit = true;");
    }
    ctr++;
    if(ctr==0)flag=1;
    lcdDisp->clear();
    lcdDisp->printf(0, 2, "%ld",ctr);
    smsg.reset(ctr, 0);
    smsg.addInt32(ctr);
    beo->send(0, smsg);
    /**************************************************/
  }

  /**************** 8: Cleanup...sort of ************/
  end=tim.get();
  elapsed=((end-start)-totalpause)/(double)1000000;
  printf("time=%f secs; processed %ld pics\n",elapsed,ctr);

  //write weights
  ffn_dm->write3L("/lab/bb/data/nnwh1dm.dat", "/lab/bb/data/nnwh2dm.dat",
    "/lab/bb/data/nnwodm.dat");
  delete ffn_dm;
  manager.stop();
  /*************************************************/

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
