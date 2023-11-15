/*!@file BeoSub/BeoSubBallast.C A Ballast for the beosub */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubBallast.C $
// $Id: BeoSubBallast.C 5132 2005-07-30 07:51:13Z itti $
//

#include "BeoSub/BeoSubBallast.H"
#include "Devices/BeoChip.H"

// ######################################################################
BeoSubBallast::BeoSubBallast(OptionManager& mgr ,
                             const std::string& descrName,
                             const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),
  itsPulsesPerFill(tagName+"PulsesPerFill", this, 435),
  itsOutRed(tagName+"OutRed", this, 0),
  itsOutWhite(tagName+"OutWhite", this, 1),
  itsInYellow(tagName+"InYellow", this, 0),
  itsInWhite(tagName+"InWhite", this, 1),
  itsBeoChip(), itsDesiredPulses(0), itsCurrentPulses(10),
  itsDirection(Idle), itsPreviousDirection(Idle),
  itsIsFull(false), itsIsEmpty(false), itsInitialized(false),
  itsPreviousInputs(0), itsGotEndstop(false)
{
  pthread_mutex_init(&itsLock, NULL);
}

// ######################################################################
void BeoSubBallast::setBeoChip(nub::soft_ref<BeoChip>& bc)
{ itsBeoChip = bc; }

// ######################################################################
BeoSubBallast::~BeoSubBallast()
{
  pthread_mutex_destroy(&itsLock);
}

// ######################################################################
void  BeoSubBallast::set(const float val, const bool blocking)
{
  if (itsBeoChip.get() == NULL) LFATAL("I need a BeoChip!");
  bool waitdone = true;

  pthread_mutex_lock(&itsLock);

  // keep track of desired value:
  float v = val;
  if (v < 0.0F) { LERROR("CAUTION: Value %f clamped to 0.0", v); v = 0.0F; }
  if (v > 1.0F) { LERROR("CAUTION: Value %f clamped to 1.0", v); v = 1.0F; }
  itsDesiredPulses = int(val * itsPulsesPerFill.getVal() + 0.5F);

  // actuate the motor:
  if (itsDesiredPulses > itsCurrentPulses) move(Filling);
  else if (itsDesiredPulses < itsCurrentPulses) move(Emptying);
  else move(Idle);

  pthread_mutex_unlock(&itsLock);

  if (blocking)
    {
      while(waitdone)
        {
          // check whether we have reached our target:
          bool done = false;
          pthread_mutex_lock(&itsLock);
          if (itsDirection == Idle) done = true;
          pthread_mutex_unlock(&itsLock);

          if (done) break; else usleep(50000);
        }
    }
}

// ######################################################################
void BeoSubBallast::mechanicalInitialize()
{
  // attempt an initialization. We don't know in which initial
  // position we may be. The following could happen:
  //
  // - we start somewhere between full and empty; then actualing the
  //   ballast in any direction should yield pulses and we may bump
  //   into an endpoint
  //
  // - we start at one endpoint; then attempting to move past it will
  //   yield no response from the BeoChip at all; moving in the other
  //   direction should yield pulses at least after a brief instant.

  CLINFO("filling...");
  set(1.0F, false); sleep(4);
  CLINFO("emptying...");
  set(0.0F, true);
}

// ######################################################################
float BeoSubBallast::get() const
{
  pthread_mutex_lock(const_cast<pthread_mutex_t *>(&itsLock));
  const float ret = float(itsCurrentPulses) / itsPulsesPerFill.getVal();
  pthread_mutex_unlock(const_cast<pthread_mutex_t *>(&itsLock));
  return ret;
}

// ######################################################################
int BeoSubBallast::getPulses() const
{
  pthread_mutex_lock(const_cast<pthread_mutex_t *>(&itsLock));
  int ret = itsCurrentPulses;
  pthread_mutex_unlock(const_cast<pthread_mutex_t *>(&itsLock));
  return ret;
}

// ######################################################################
void BeoSubBallast::input(const int val)
{
  // use alternate input() function during init:
  if (itsInitialized == false) { inputDuringInit(val); return; }

  pthread_mutex_lock(&itsLock);

  // otherwise let's decode the input:
  int changed = itsPreviousInputs ^ val;
  bool opto = ((changed & (1 << itsInYellow.getVal())) != 0);
  bool endstop = ((val & (1 << itsInWhite.getVal())) != 0);
  itsPreviousInputs = val;

  //CLINFO("Input received: opto=%d, endstop=%d, dir=%d, pdir=%d",
  //       int(opto), int(endstop), int(itsDirection),
  //       int(itsPreviousDirection));

  // did we just receive a pulse from the opto encoder? if so,
  // increment/decrement our pulse counter according to our current
  // direction of motion. If we are not moving, probably we just
  // decided to stop the motors but they are still spinning in
  // whatever previous motion direction:
  if (opto)
    {
      switch (itsDirection)
        {
        case Filling:
          ++ itsCurrentPulses;
          if (itsCurrentPulses >= itsDesiredPulses) move(Idle);
          break;

        case Emptying:
          -- itsCurrentPulses;
          if (itsCurrentPulses <= itsDesiredPulses) move(Idle);
          break;

        case Idle:
          {
            switch (itsPreviousDirection)
              {
              case Filling:
                if (itsCurrentPulses < itsPulsesPerFill.getVal())
                  ++ itsCurrentPulses;
                break;

              case Emptying:
                if (itsCurrentPulses > 0)
                  -- itsCurrentPulses;
                break;

              case Idle:
                CLERROR("Received pulse while Idle! -- DISCARDED");
              }
          }
        }
    }

  // clear our endstop history if appropriate:
  if (endstop == false) itsGotEndstop = false;

  // did we just hit an endpoint? If so, we can reset our pulse counter:
  // NOTE: endstop signals only work when we apply current to the motor...
  if (itsDirection != Idle)
    {
      if (endstop == false)
        {
          itsIsFull = false;
          itsIsEmpty = false;
        }
      else
        {
          // we just hit an endpoint; first, were we in travel (this
          // is to implement hysteresis at the endpoints - we wait
          // until we get out of an endpoint before we consider
          // hitting it again)?
          if (itsIsFull == false && itsIsEmpty == false)
            {
              // if we already got a first endstop pulse, this is now
              // the second one, and we will accept it no matter
              // what. To know which endstop we hit, we just pick the
              // one we are closest to. Note that we used to take our
              // current motion direction into account (Filling or
              // Emptying) but that may be bogus if we are flipping
              // directions quickly (the inertia of the motor may
              // trick us). Now, could this be a spurious endstop
              // pulse (if it's the first one)? If so, we will only
              // get one, while we usually get two when we truly hit
              // the endpoint (but this is not guaranteed, so we will
              // still accept a single pulse if we are reasonably
              // close to an endpoint according to our pulse counter):
              if (itsGotEndstop ||
                  itsCurrentPulses < itsPulsesPerFill.getVal() / 10 ||
                  itsCurrentPulses > (itsPulsesPerFill.getVal()*9) / 10)
                {
                  // which side of the middle point are we?
                  if (itsCurrentPulses > itsPulsesPerFill.getVal() / 2)
                    {
                      // ok so we must now be full:
                      itsCurrentPulses = itsPulsesPerFill.getVal();
                      move(Idle); itsIsFull = true;
                      CLINFO("Endstop reached -> full");
                    }
                  else
                    {
                      // ok so we must now be empty:
                      itsCurrentPulses = 0;
                      move(Idle); itsIsEmpty = true;
                      CLINFO("Endstop reached -> empty");
                    }
                }
              else
                itsGotEndstop = true; // fishy; wait for second one
            }
        }
    }

  //CLINFO("desired=%d current=%d", itsDesiredPulses, itsCurrentPulses);

  pthread_mutex_unlock(&itsLock);
}

// ######################################################################
void BeoSubBallast::inputDuringInit(const int val)
{
  pthread_mutex_lock(&itsLock);

  bool endstop = ((val & (1 << itsInWhite.getVal())) != 0);
  itsPreviousInputs = val;

  //CLINFO("Input received during init: endstop=%d", int(endstop));

  // boot sequence should be to fill for a while, then empty until we
  // hit the endpoint. Once we are completely empty, we will turn
  // itsInitialized to true.

  // did we just hit an endpoint? If so, we can reset our pulse counter:
  // NOTE: endstop signals only work when we apply current to the motor...
  if (itsDirection != Idle)
    {
      if (endstop == false)
        {
          itsIsFull = false;
          itsIsEmpty = false;
        }
      else
        {
          // we just hit an endpoint:
          switch (itsDirection)
            {
            case Filling:
              // were we in travel?
              if (itsIsFull == false && itsIsEmpty == false)
                {
                  // ok so we must now be full:
                  itsCurrentPulses = itsPulsesPerFill.getVal();
                  move(Idle); itsIsFull = true;
                }
              break;

            case Emptying:
              // were we in travel?
              if (itsIsFull == false && itsIsEmpty == false)
                {
                  // ok so we must now be empty:
                  itsCurrentPulses = 0;
                  move(Idle); itsIsEmpty = true;


                  // ok, we are now initialized:
                  CLINFO("Mechanical initialization complete.");
                  itsInitialized = true;
                }
              break;

            default:
              break;
            }
        }
    }

  //CLINFO("desired=%d current=%d", itsDesiredPulses, itsCurrentPulses);
  pthread_mutex_unlock(&itsLock);
}

// ######################################################################
bool BeoSubBallast::moving() const
{
  pthread_mutex_lock(const_cast<pthread_mutex_t *>(&itsLock));
  bool ret = (itsDirection != Idle);
  pthread_mutex_unlock(const_cast<pthread_mutex_t *>(&itsLock));
  return ret;
}

// ######################################################################
void BeoSubBallast::move(const MotorDirection dir)
{
  // CAUTION: Should only be called under protection of our itsLock!
  switch(dir)
    {
    case Filling:
      if (itsDirection != Filling) itsPreviousDirection = itsDirection;
      itsDirection = Filling;
      itsBeoChip->setDigitalOut(itsOutRed.getVal(), true);
      itsBeoChip->setDigitalOut(itsOutWhite.getVal(), false);
      break;
    case Emptying:
      if (itsDirection != Emptying) itsPreviousDirection = itsDirection;
      itsDirection = Emptying;
      itsBeoChip->setDigitalOut(itsOutRed.getVal(), false);
      itsBeoChip->setDigitalOut(itsOutWhite.getVal(), true);
      break;
    case Idle:
      if (itsDirection != Idle) itsPreviousDirection = itsDirection;
      itsDirection = Idle;
      itsBeoChip->setDigitalOut(itsOutRed.getVal(), false);
      itsBeoChip->setDigitalOut(itsOutWhite.getVal(), false);
      break;
    default:
      CLERROR("Unknown move command ignored");
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
