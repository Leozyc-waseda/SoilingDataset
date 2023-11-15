/*
 * OptiSync
 *
 * Dual engine syncronizer for Mercury Optimax outboards.
 * 
 * Copyright (C) 2009 by Laurent Itti. All rights reserved.
 *
 */

#include <ks0108iLab.h>    // graphic LCD routines
#include <MsTimer2.h>

#include <Arial12.h>       // font definitions 
#include <Impact28.h>      // font definitions 
#include <Futura40.h>      // font definitions 
#include <Font5x7.h>

// 225HP Mercury Optimax outboard (3.0L 6-cyl DFI) = 6 pulses per revolution (aka 12 poles)
#define PPR 6

// Max RPM is around 6000. That's 100 revolutions/s, i.e., 600 pulses/s at the most
// Idle RPM is at 550. That's about 9 revolutions/s, i.e., should expect no less than 50 pulses/s
// If we count pulses over 250ms we expect 10 - 150 pulses each cycle, fits in a byte.
#define TMRLOOP 250
volatile byte portcount;
volatile byte stbdcount;

// number of averages for frequency measurements (in multiples of timer2 calls)
#define NAVG 20
byte portdata[NAVG];
byte stbddata[NAVG];
byte dataidx;

// flag to refresh display
volatile boolean refresh;

// computed RPM
volatile uint16_t portrpm;
volatile uint16_t stbdrpm;

// PIN of blinking LED (LED is built-in on Arduino Pro Mini)
#define LEDPIN 13

// interrupt number for port tach (corresponds to PIN 2)
#define PORTINT 0
#define PORTPIN 2

// interrupt number for starboard tach (corresponds to PIN 3)
#define STBDINT 1
#define STBDPIN 3

// led blinking toggle
boolean ledon;

// Interrupt routine for port tach
void portInt() { ++ portcount; }

// Interrupt routine for starboard tach
void stbdInt() { ++ stbdcount; }

// Interrupt every 250ms from Timer2
void timerInt() {
  // get the latest data and reset port and starboard counters:
  portdata[dataidx] = portcount; stbddata[dataidx] = stbdcount;
  portcount = 0; stbdcount = 0;
  ++dataidx; if (dataidx >= NAVG) dataidx = 0;
  
  // compute average; first get total pulses per buffer period
  portrpm = 0; stbdrpm = 0;
  for (byte i = 0; i < NAVG; ++i) { portrpm += (uint16_t)(portdata[i]); stbdrpm += (uint16_t)(stbddata[i]); }
  
  portrpm *= (60000 / (NAVG * TMRLOOP * PPR));  // total revolutions per minute
  stbdrpm *= (60000 / (NAVG * TMRLOOP * PPR));  // total revolutions per minute
  
  if (ledon) digitalWrite(LEDPIN, LOW); else digitalWrite(LEDPIN, HIGH);
  ledon = ! ledon;
  
  // instruct our main loop that it's time to refresh:
  refresh = true;
}

// ##################################################################################
void setup() {
  unsigned long startMillis = millis();
  portcount = 0; stbdcount = 0; portrpm = 0; stbdrpm = 0;
  dataidx = 0; refresh = false; ledon = false;
  pinMode(LEDPIN, OUTPUT);
  pinMode(PORTPIN, INPUT);
  pinMode(STBDPIN, INPUT);
  attachInterrupt(PORTINT, portInt, RISING);
  attachInterrupt(STBDINT, stbdInt, RISING);
  for (int i = 0; i < NAVG; ++i) { portdata[i] = 0; stbddata[i] = 0; }

  MsTimer2::set(TMRLOOP, timerInt);
  MsTimer2::start();
  
  GLCDiLab.Init(NON_INVERTED);   // initialize the library, non inverted writes pixels onto a clear screen
  GLCDiLab.ClearScreen();
  GLCDiLab.SelectFont(Futura40);
  GLCDiLab.GotoXY(2, 0);
  GLCDiLab.Puts("OptiSync");
  GLCDiLab.SelectFont(Arial12);
  GLCDiLab.GotoXY(3, 48);
  GLCDiLab.Puts("Dual  Engine  Synchronizer");
  while (millis() - startMillis < 5000) { }
  GLCDiLab.ClearScreen();               // clear the screen  
}

// ##################################################################################
void  loop() {   // run over and over again
  // wait for timer2 interrupt:
  while (refresh == false) { }
  long prpm = (long)portrpm; long srpm = (long)stbdrpm;
  refresh = false; // get ready to handle next interrupt
  
  if (prpm > 9999) prpm = 9999;
  if (srpm > 9999) srpm = 9999;
  
  // refresh the display:
  GLCDiLab.SelectFont(Impact28); // you can also make your own fonts, see playground for details  

  GLCDiLab.PrintNumberCentered(prpm, 0, 0, 64);
  GLCDiLab.PrintNumberCentered(srpm, 64, 0, 64);
  
  long diff = prpm - srpm;

  GLCDiLab.SelectFont(Font5x7);
  GLCDiLab.PrintNumberCentered(diff >= 0 ? diff : -diff, 46, 32, 36);

  // useful range for bargraph is +/- 60, let's show +/-180 RPM:
  diff /= 3; if (diff > 60) diff = 60; else if (diff < -60) diff = -60;
  GLCDiLab.DrawBarGraph(48, int8_t(diff));
  GLCDiLab.FillRect(63, 44, 2, 20, BLACK);
}
