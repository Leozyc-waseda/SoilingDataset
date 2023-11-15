/*!@file Parallel/SingleChannelBeoServer.C server to work with SingleChannelBeo */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/SingleChannelBeoServer.C $
// $Id: SingleChannelBeoServer.C 8264 2007-04-17 21:43:10Z rjpeters $
//

#include "Parallel/SingleChannelBeoServer.H"

#include "Beowulf/Beowulf.H"
#include "Beowulf/BeowulfOpts.H"
#include "Channels/BlueYellowChannel.H"
#include "Channels/ColorBandChannel.H"
#include "Channels/DirectionChannel.H"
#include "Channels/FlickerChannel.H"
#include "Channels/GaborChannel.H"
#include "Channels/IntensityChannel.H"
#include "Channels/RedGreenChannel.H"
#include "Channels/SingleChannel.H"
#include "Component/OptionManager.H"
#include "Component/ParamMap.H"
#include "Image/PyramidTypes.H"
#include "Neuro/SingleChannelBeo.H"    // for the BEO_XXX message IDs
#include "Neuro/VisualCortexSurprise.H"
#include "Util/Assert.H"
#include "Util/sformat.H"

#include <sstream>

// ######################################################################
SingleChannelBeoServer::SingleChannelBeoServer(OptionManager& mgr,
                                               const std::string& descrName,
                                               const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),
  itsQuickMode(&OPT_SingleChannelBeoServerQuickMode, this),
  itsBeo(new Beowulf(mgr))
{
  // get our Beowulf as a subcomponent:
  addSubComponent(itsBeo);

  VisualCortexSurprise::registerSurpriseTypes(mgr);
}

// ######################################################################
SingleChannelBeoServer::~SingleChannelBeoServer()
{  }

// ######################################################################
void SingleChannelBeoServer::check()
{
  // See if we have a new message on our Beowulf:
  TCPmessage rmsg;
  int32 rframe, raction, rnode = -1;  // receive from any node

  if (itsBeo->receive(rnode, rmsg, rframe, raction, 50)) // wait up to 50ms
    switch(raction) {
    case BEO_INIT:       // ########## Beowulf reset
      {
        // ooops, someone wants to re-initialize us!
        removeAllSubComponents(); // kill all our channels
        // reinitialization of beowulf is handled automatically.
      }
      break;
      case BEO_SCHANCONF:  // ########## want to configure a new channel
      {
        // get the VisualFeature, channel index (if complex channel)
        // and the two ParamMaps out of the message:
        VisualFeature vs = static_cast<VisualFeature>(rframe);
        ParamMap params, coeffs;

        const std::string dnam = rmsg.getElementString();
        const std::string tnam = rmsg.getElementString();

        std::string buf = rmsg.getElementString();
        std::stringstream sparam(buf);
        params.load(sparam);

        buf = rmsg.getElementString();
        std::stringstream scoeff(buf);
        coeffs.load(scoeff);

        // do we already have a channel for this node? if so, destroy:
        nub::soft_ref<SingleChannel> ch = channel(rnode);
        if (ch.isValid())
          {
            LINFO("Deleting '%s' of node %d", ch->descriptiveName().c_str(),
                  rnode);
            ch->stop();
            removeSubComponent(*ch);
            ch.reset(NULL);
          }

        // let's instantiate the channel according to the
        // VisualFeature and descr/tag names. NOTE: for those that
        // take extra constructor args (e.g., OrientationChannel), we
        // put dummy default values here; they will be re-loaded from
        // the ParamMap anyway:
        switch(vs)
          {
          case INTENS:  ch.reset(new IntensityChannel(getManager())); break;
          case BY:      ch.reset(new BlueYellowChannel(getManager())); break;
          case RG:      ch.reset(new RedGreenChannel(getManager())); break;
          case FLICKER: ch.reset(new FlickerChannel(getManager())); break;
          case COLBAND: ch.reset(new ColorBandChannel(getManager(), 0)); break;
          case ORI:     ch.reset(new GaborChannel(getManager(),0, 0.0)); break;
          case MOTION:  ch.reset(new DirectionChannel(getManager(), 0, 0.0,
                                                      Gaussian9)); break;
          default:      LFATAL("Unsupported VisualFeature %d", rframe);
          }

        // set our channel's descriptive and tag names:
        ch->setDescriptiveName(dnam);
        ch->setTagName(tnam);

        // let's configure the channel (we we need the correct tagName
        // for this to work, which is why we got it in our message):
        ch->readParamsFrom(params, false); // generate errors if params wrong

        // let's now rename the channel with the requester's Beowulf
        // node number, so that we can later find it back using our
        // channel() function:
        ch->setTagName(sformat("%d", rnode));

        // let's start the channel:
        ch->start();

        // let's load the channel's submap coefficients:
        ch->readFrom(coeffs);

        // let's add the channel as one of our subComponents:
        addSubComponent(ch);

        // ready for some channel action!
        LINFO("Created '%s' for %s [node %d]",
              ch->descriptiveName().c_str(), itsBeo->nodeName(rnode), rnode);
      }
      break;
    case BEO_SCHANINPUT: // ########## some input to process
      {
        // do we have a channel for this node?
        nub::soft_ref<SingleChannel> ch = channel(rnode);
        if (ch.isInvalid())
          LERROR("Input from node %d but no channel for it -- IGNORING",rnode);
        else
          {
            Timer tim(1000000);  // let's keep track of how long we process
            Dims dims;           // let's also report input size

            // let's decode the message, do the processing and send
            // the output back:
            const double t = rmsg.getElementDouble();
            Image<float> im = rmsg.getElementFloatIma();
            Image<byte> cm = rmsg.getElementByteIma();
            ch->input(InputFrame::fromGrayFloat(&im, SimTime::SECS(t), &cm));
            dims = im.getDims();

            // send our results back: either just time and output (if
            // quick mode), or time, output, pyramid, submaps and
            // clipPyr (if not quick mode):
            TCPmessage msg(rframe, BEO_SCHANOUTPUT);
            if (itsQuickMode.getVal() == false)
              msg.reset(rframe, BEO_SCHANALLOUT);

            msg.addDouble(t);

            // getOutput() always works, possibly returning a blank
            // image if we have no output available:
            msg.addImage(ch->getOutput());

            if (itsQuickMode.getVal() == false) {
              // for the other internal maps, if we have an input
              // pyramid, then send the maps back, otherwise send
              // empty ImageSets back:
              if (ch->hasPyramid()) {
                msg.addImageSet(ch->pyramid(0));

                uint ns = ch->numSubmaps(); ImageSet<float> submaps(ns);
                for (uint i = 0; i < ns; i ++) submaps[i] = ch->getSubmap(i);
                msg.addImageSet(submaps);

                msg.addImageSet(ch->clipPyramid());
              } else {
                ImageSet<float> empty;
                msg.addImageSet(empty);  // the pyramid
                msg.addImageSet(empty);  // the submaps
                msg.addImageSet(empty);  // the clipPyr
              }
            }

            // send the results to our master:
            itsBeo->send(rnode, msg);
            LINFO("%sProcessed %dx%d input %d in %lluus",
                  itsQuickMode.getVal()?"Quick":"", dims.w(), dims.h(),
                  rframe, tim.get());
          }
      }
      break;
    default:
      LERROR("Received unknown message from node %d -- IGNORING", rnode);
    }
}

// ######################################################################
nub::soft_ref<SingleChannel> SingleChannelBeoServer::channel(const int32 node)
{
  // convention used here: our channels have as tag name the number of
  // the node that requested them:
  const std::string name = sformat("%d", node);

  nub::soft_ref<SingleChannel> chan;
  if (hasSubComponent(name)) dynCastWeakToFrom(chan, subComponent(name));

  return chan;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
