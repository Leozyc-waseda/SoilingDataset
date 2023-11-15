/*!@file Parallel/pvisionTCP2.C A parallel vision slave to use w/ pvisionTCP2master */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/pvisionTCP2.C $
// $Id: pvisionTCP2.C 8264 2007-04-17 21:43:10Z rjpeters $
//

/*! See the pvisionTCP2go script in bin/ for how to launch the slaves, and
  see pvisionTCP2-master.C for the master program.
*/

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Parallel/pvisionTCP-defs.H"
#include "Util/Assert.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/sformat.H"

#include <cstdlib>
#include <signal.h>
#include <time.h>
#include <unistd.h>

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; }

//! Compute a conspicuity map from an image received in a message
void computeCMAP(TCPmessage& msg, const PyramidType ptyp,
                 const float ori, const float coeff,
                 const int slave, nub::soft_ref<Beowulf>& b);

//! Compute a conspicuity map from two images received in a message
void computeCMAP2(TCPmessage& msg, const PyramidType ptyp,
                 const float ori, const float coeff,
                 const int slave, nub::soft_ref<Beowulf>& b);

//! Compute a conspicuity map from an image
void computeCMAP(const Image<float>& fima, const PyramidType ptyp,
                 const float ori, const float coeff,
                 const int slave, nub::soft_ref<Beowulf>& b, const int32 id);

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

// number of output saliency maps kept in buffer (to accomodate for
// speed variations):
#define NBOUT 10

// relative feature weights:
#define IWEIGHT 1.0
#define CWEIGHT 1.0
#define OWEIGHT 1.0
#define FWEIGHT 1.5

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Parallel Vision TCP Version 2 - Slave");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  initRandomNumbers();

  // various processing inits:
  TCPmessage rmsg;            // message being received and to process
  TCPmessage smsg;            // message being sent
  Image<float> outmap[NBOUT]; // array of output saliency maps
  int32 outframe[NBOUT];      // array of output frame numbers
  int nbcmap[NBOUT];          // how many consp maps in each outmap
  for (int i = 0; i < NBOUT; i ++) { outframe[i] = -1; nbcmap[i] = 0; }
  Image<float> previma;       // previous image; for flicker processing
  Timer tim;                  // to measure processing speed

  // let's get all our ModelComponent instances started:
  manager.start();

  // wait for data and process it:
  while(goforever)
    {
      int32 rframe, raction, rnode = -1;  // receive from any node
      if (beo->receive(rnode, rmsg, rframe, raction, 3))  // wait up to 3ms
        {
          //LINFO("Frame %d, action %d from node %d", rframe, raction, rnode);
          tim.reset();

          // select processing branch based on frame number:
          switch(raction)
            {
            case BEO_INIT:       // ##############################
              {
                // ooops, someone wants to re-initialize us!
                // forget about all our current frames:
                for (int i = 0; i < NBOUT; i ++)
                  { outframe[i] = -1; nbcmap[i] = 0; }
                // reinitialization of beowulf is handled automatically.
              }
              break;
            case BEO_RETINA:     // ##############################
              {
                // get the color image:
                Image<PixRGB<byte> > ima = rmsg.getElementColByteIma();

                // compute luminance and send it off:
                Image<byte> lum = luminance(ima);
                smsg.reset(rframe, BEO_LUMINANCE);
                smsg.addImage(lum);
                beo->send(0, smsg);  // send off to luminance slave

                // compute RG and BY and send them off:
                Image<byte> r, g, b, y; getRGBY(ima, r, g, b, y, (byte)25);
                smsg.reset(rframe, BEO_REDGREEN);
                smsg.addImage(r); smsg.addImage(g);
                beo->send(1, smsg);  // send off to RG slave
                smsg.reset(rframe, BEO_BLUEYELLOW);
                smsg.addImage(b); smsg.addImage(y);
                beo->send(2, smsg);  // send off to BY slave
              }
              break;
            case BEO_LUMINANCE:  // ##############################
              {
                // first, send off luminance to orientation slaves:
                rmsg.setAction(BEO_ORI0);
                beo->send(3, rmsg);  // send off to ori0 slave
                rmsg.setAction(BEO_ORI45);
                beo->send(4, rmsg);  // send off to ori45 slave
                rmsg.setAction(BEO_ORI90);
                beo->send(5, rmsg);  // send off to ori90 slave
                rmsg.setAction(BEO_ORI135);
                beo->send(6, rmsg);  // send off to ori135 slave

                // and also send to flicker slave:
                rmsg.setAction(BEO_FLICKER);
                beo->send(8, rmsg);  // send off to flick slave

                // now compute intensity conspicuity map and send to collector:
                computeCMAP(rmsg, Gaussian5, 0.0, IWEIGHT, 7, beo);
              }
              break;
            case BEO_FLICKER:
              {
                // get the luminance image out of the message:
                Image<byte> ima = rmsg.getElementByteIma();
                Image<float> fima = ima;  // convert to float

                // compute flicker consp map and send to collector:
                if (previma.initialized() == false) previma = fima;
                previma -= fima;
                computeCMAP(previma, Gaussian5, 0.0, FWEIGHT, 7,
                            beo, rframe);
                previma = fima;
              }
              break;
            case BEO_REDGREEN:   // ##############################
              computeCMAP2(rmsg, Gaussian5, 0.0, CWEIGHT, 7, beo);
              break;
            case BEO_BLUEYELLOW: // ##############################
              computeCMAP2(rmsg, Gaussian5, 0.0, CWEIGHT, 7, beo);
              break;
            case BEO_ORI0:       // ##############################
              computeCMAP(rmsg, Oriented5, 0.0, OWEIGHT, 7, beo);
              break;
            case BEO_ORI45:      // ##############################
              computeCMAP(rmsg, Oriented5, 45.0, OWEIGHT, 7, beo);
              break;
            case BEO_ORI90:      // ##############################
              computeCMAP(rmsg, Oriented5, 90.0, OWEIGHT, 7, beo);
              break;
            case BEO_ORI135:     // ##############################
              computeCMAP(rmsg, Oriented5, 135.0, OWEIGHT, 7, beo);
              break;
            case BEO_CMAP:       // ##############################
              {
                // get the map:
                Image<float> ima = rmsg.getElementFloatIma();

                // do we already have some accumulated data on this
                // frame, or is this a new frame?
                int32 oldestidx = 0;
                int32 oldestframe = std::numeric_limits<int32>::max();
                int32 curridx = -1;
                for (int32 i = 0; i < NBOUT; i ++)
                  {
                    if (outframe[i] < oldestframe)
                      { oldestframe = outframe[i]; oldestidx = i; }
                    if (outframe[i] == rframe) // ok, already in progress
                      { curridx = i; break; }
                  }

                // if we are not already processing this frame, set it up,
                // possibly dropping oldest frame in buffer if buffer full:
                if (curridx == -1)
                  {
                    // ok, use oldest frame, then...
                    curridx = oldestidx;
                    // was it in progress?
                    if (nbcmap[curridx])
                      {
                        LINFO("Dropping frame %d!", outframe[curridx]);
                        std::string junk;
                        for (int i = 0; i < NBOUT; i ++)
                          junk += sformat("%d/%d ", outframe[i], nbcmap[i]);

                        LINFO("BUFFER = [ %s]", junk.c_str());
                      }
                    // now we process frame nb rframe at slot curridx:
                    outframe[curridx] = rframe; nbcmap[curridx] = 0;
                    outmap[curridx].resize(ima.getWidth(),
                                           ima.getHeight(), true);
                  }

                // add received cmap to output map:
                outmap[curridx] += ima; nbcmap[curridx] ++;

                // have we received all the cmaps?
                if (nbcmap[curridx] == NBCMAP2)
                  {
                    outmap[curridx] =
                      maxNormalize(outmap[curridx], 0.0f, 9.0f, normtyp);
                    inplaceClamp(outmap[curridx], 0.0f, 255.0f);

                    // send result to master:
                    smsg.reset(outframe[curridx], BEO_WINNER);
                    Image<byte> smap = outmap[curridx]; // convert to byte

                    smsg.addImage(smap);
                    beo->send(-1, smsg);
                    nbcmap[curridx] = 0;  // ready for next frame
                    //LINFO("sending off %d", outframe[curridx]);
                  }
              }
              break;
            default: // ##############################
              LERROR("Bogus action %d -- IGNORING.", raction);
              break;
            }
          if (rframe % 101 == 0)
            LINFO("Action %d, frame %d processed in %llums",
                  raction, rframe, tim.get());
        }
    }

  // we got broken:
  manager.stop();
  return 0;
}

// ######################################################################
void computeCMAP(TCPmessage& msg, const PyramidType ptyp,
                 const float ori, const float coeff,
                 const int slave, nub::soft_ref<Beowulf>& b)
{
  Image<byte> ima = msg.getElementByteIma();
  Image<float> fima = ima; // convert to float

  computeCMAP(fima, ptyp, ori, coeff, slave, b, msg.getID());
}

// ######################################################################
void computeCMAP2(TCPmessage& msg, const PyramidType ptyp,
                 const float ori, const float coeff,
                 const int slave, nub::soft_ref<Beowulf>& b)
{
  Image<byte> ima1 = msg.getElementByteIma();
  Image<byte> ima2 = msg.getElementByteIma();
  Image<float> fima = ima1 - ima2;

  computeCMAP(fima, ptyp, ori, coeff, slave, b, msg.getID());
}

// ######################################################################
void computeCMAP(const Image<float>& fima, const PyramidType ptyp,
                 const float ori, const float coeff,
                 const int slave, nub::soft_ref<Beowulf>& b, const int32 id)
{
  // compute pyramid:
  ImageSet<float> pyr = buildPyrGeneric(fima, 0, maxdepth, ptyp, ori);

  // alloc conspicuity map and clear it:
  Image<float> cmap(pyr[sml].getDims(), ZEROS);

  // intensities is the max-normalized weighted sum of IntensCS:
  for (int delta = delta_min; delta <= delta_max; delta ++)
    for (int lev = level_min; lev <= level_max; lev ++)
      {
        Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
        tmp = downSize(tmp, cmap.getWidth(), cmap.getHeight());
        inplaceAddBGnoise(tmp, 255.0);
        tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
        cmap += tmp;
      }
  if (normtyp == VCXNORM_MAXNORM)
    cmap = maxNormalize(cmap, MAXNORMMIN, MAXNORMMAX, normtyp);
  else
    cmap = maxNormalize(cmap, 0.0f, 0.0f, normtyp);

  // multiply by conspicuity coefficient:
  cmap *= coeff;

  //float mi, ma; getMinMax(cmap, mi, ma); LINFO("[%f .. %f]", mi, ma);

  // send off resulting conspicuity map:
  TCPmessage smsg(id, BEO_CMAP);
  smsg.addImage(cmap);
  b->send(slave, smsg);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
