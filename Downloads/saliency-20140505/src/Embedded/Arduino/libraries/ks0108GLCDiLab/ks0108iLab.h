

/* This class was modified by Laurent Itti from the following.
   Modifications include:
   - timing to work on a 16MHz arduino
   - new bargraph graphics
   - fix code throughout so that when a width and height are used, they
     are the actual drawn width and height (as opposed to width-1 and
     height-1). So FillRect(0, 0, 128, 164, BLACK) fills the entire
     128x64 screen.
*/


/*
  ks0108.h - Arduino library support for ks0108 and compatable graphic LCDs
  Copyright (c)2008 Michael Margolis All right reserved
  mailto:memargolis@hotmail.com?subject=KS0108_Library 

  This library is based on version 1.1 of the excellent ks0108 graphics routines written and
  copyright by Fabian Maximilian Thiele. His sitelink is  
  dead but you can obtain a copy of his original work here:
  http://www.scienceprog.com/wp-content/uploads/2007/07/glcd_ks0108.zip

  Code changes include conversion to an Arduino C++ library, adding more
  flexibility in port addressing and improvements in I/O speed. The interface 
  has been made more Arduino friendly and some convenience functions added. 

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

  Version:   1.0  - May 8 2008 - initial release
  Version:   1.0a - Sept 1 2008 - simplified command pin defines  
  Version:   1.0b - Sept 18 2008 - replaced <wiring.h> with boolean typedef for rel 0012  
*/

#include <inttypes.h>
//#include <wiring.h> // for boolean
typedef uint8_t boolean;
typedef uint8_t byte;
#include <avr/pgmspace.h>

#ifndef        KS0108ILAB_H
#define KS0108ILAB_H

/*********************************************************/
/*  Configuration for assigning LCD bits to Arduino Pins */
/*********************************************************/
/* Arduino pins used for Commands
 * default assignment uses the first five analog pins
 */

#define CSEL1 14  // CS1 Bit   // swap pin assignments with CSEL2 if left/right image is reversed
#define CSEL2 15  // CS2 Bit
#define R_W   16  // R/W Bit
#define D_I   17  // D/I Bit 
#define EN    18  // EN Bit

/* Arduino pins used for LCD Data 
 * un-comment ONE of the following pin options that corresponds to the wiring of data bits 0-3 
 */
#define dataPins8to11   // bits 0-3 assigned to arduino pins 8-11, bits 4-7 assigned to arduino pins 4-7
//#define dataPins14to17 //bits 0-3 assigned to arduino pins 14-17, bits 4-7 assigned to arduino pins 4-7. (note command pins must be changed)
//#define dataPins0to3  // bits 0-3 assigned to arduino pins 0-3 , bits 4-7 assigned to arduino pins 4-7, this is marginally  the fastest option but  its only available on runtime board without hardware rs232.

/* NOTE: all above options assume LCD data bits 4-7 are connected to arduino pins 4-7 */
/*******************************************************/
/*     end of Arduino pin configuration                */
/*******************************************************/

/* option: uncomment the next line if all command pins are on the same port for slight speed & code size improvement */
//#define LCD_CMD_PORT                PORTC                // Command Output Register 

//#define HD44102   // uncomment this to build a 44102 version

#ifndef dataPins0to3                     // this is the only option on standard arduino where all data bits are on same port 
#define LCD_DATA_NIBBLES                // if this is defined then data i/o is split into two operations
#endif 

// these macros  map pins to ports using the defines above
// the following should not be changed unless you really know what your doing 
#ifdef dataPins0to3
#define LCD_DATA_LOW_NBL   D   // port for low nibble: D=pins 0-3  
#endif
#ifdef dataPins14to17 
#define LCD_DATA_LOW_NBL   C   // port for low nibble: C=pins 14-17 (using this requires reasignment of command pins) 
#endif
#ifdef dataPins8to11            // the following is the defualt setting
#define LCD_DATA_LOW_NBL   B   // port for low nibble, B=pins 8-11
#endif

#define LCD_DATA_HIGH_NBL  D   // port for high nibble: D=pins 4-7, B & C not available on std arduino  

// macros for pasting port defines
#define GLUE(a, b)     a##b 
#define PORT(x)        GLUE(PORT, x)
#define PIN(x)         GLUE(PIN, x)
#define DDR(x)         GLUE(DDR, x)

// paste together the port definitions if using nibbles
#define LCD_DATA_IN_LOW                PIN(LCD_DATA_LOW_NBL)        // Data I/O Register, low nibble
#define LCD_DATA_OUT_LOW        PORT(LCD_DATA_LOW_NBL)  // Data Output Register - low nibble
#define LCD_DATA_DIR_LOW        DDR(LCD_DATA_LOW_NBL)        // Data Direction Register for Data Port, low nibble

#define LCD_DATA_IN_HIGH        PIN(LCD_DATA_HIGH_NBL)        // Data Input Register  high nibble
#define LCD_DATA_OUT_HIGH        PORT(LCD_DATA_HIGH_NBL)        // Data Output Register - high nibble
#define LCD_DATA_DIR_HIGH        DDR(LCD_DATA_HIGH_NBL)        // Data Direction Register for Data Port, high nibble

#define lcdDataOut(_val_) LCD_DATA_OUT(_val_) 
#define lcdDataDir(_val_) LCD_DATA_DIR(_val_) 

// macros to handle data output
#ifdef LCD_DATA_NIBBLES  // data is split over two ports 
#define LCD_DATA_OUT(_val_) \
    LCD_DATA_OUT_LOW =  (LCD_DATA_OUT_LOW & 0xF0)| (_val_ & 0x0F); LCD_DATA_OUT_HIGH = (LCD_DATA_OUT_HIGH & 0x0F)| (_val_ & 0xF0); 

#define LCD_DATA_DIR(_val_)\
    LCD_DATA_DIR_LOW =  (LCD_DATA_DIR_LOW & 0xF0)| (_val_ & 0x0F); LCD_DATA_DIR_HIGH = (LCD_DATA_DIR_HIGH & 0x0F)| (_val_ & 0xF0);
#else  // all data on same port (low equals high)
#define LCD_DATA_OUT(_val_) LCD_DATA_OUT_LOW = (_val_);                
#define LCD_DATA_DIR(_val_) LCD_DATA_DIR_LOW = (_val_);
#endif


// macros to fast write data to pins known at compile time, this is over 30 times faster than digitalWrite
#define fastWriteHigh(_pin_) ( _pin_ < 8 ?  PORTD |= 1 << (_pin_ & 0x07) : ( _pin_ < 14 ?  PORTB |= 1 << ((_pin_ -8) & 0x07) : PORTC |= 1 << ((_pin_ -14) & 0x07)  ) ) 
#define fastWriteLow(_pin_) ( _pin_ < 8 ?   PORTD &= ~(1 << (_pin_  & 0x07)) : ( _pin_ < 14 ?  PORTB &= ~(1 << ((_pin_ -8) & 0x07) )  :  PORTC &= ~(1 << ((_pin_ -14) & 0x07) )  ) )

// Chips
#define CHIP1                                0x00
#define CHIP2                                0x01
#ifdef HD44102 
#define CHIP_WIDTH          50          // pixels per chip
#else
#define CHIP_WIDTH          64 
#endif

// Commands
#ifdef HD44102 
#define LCD_ON                                0x39
#define LCD_OFF                                0x38
#define LCD_DISP_START                0x3E   // Display start page 0
#else
#define LCD_ON                                0x3F
#define LCD_OFF                                0x3E
#define LCD_DISP_START                0xC0
#endif

#define LCD_SET_ADD                        0x40
#define LCD_SET_PAGE                0xB8


// Colors
#define BLACK                                0xFF
#define WHITE                                0x00

// useful user contants
#define NON_INVERTED false
#define INVERTED     true

// Font Indices
#define FONT_LENGTH                        0
#define FONT_FIXED_WIDTH        2
#define FONT_HEIGHT                        3
#define FONT_FIRST_CHAR                4
#define FONT_CHAR_COUNT                5
#define FONT_WIDTH_TABLE        6

#ifdef HD44102 
#define DISPLAY_WIDTH 100
#define DISPLAY_HEIGHT 32
#else
#define DISPLAY_WIDTH 128
#define DISPLAY_HEIGHT 64
#endif

// Uncomment for slow drawing
// #define DEBUG

typedef struct {
        uint8_t x;
        uint8_t y;
        uint8_t page;
} lcdCoord;

typedef uint8_t (*FontCallback)(const uint8_t*);

uint8_t ReadFontData(const uint8_t* ptr);        //Standard Read Callback

#define DrawVertLine(x, y, length, color) FillRect(x, y, 1, length, color)
#define DrawHoriLine(x, y, length, color) FillRect(x, y, length, 1, color)
#define DrawCircle(xCenter, yCenter, radius, color) DrawRoundRect(xCenter-radius, yCenter-radius, 2*radius, 2*radius, radius, color)
#define ClearScreen() FillRect(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, WHITE)

class ks0108iLab  // shell class for ks0108iLab glcd code
{
 public:
  ks0108iLab();
  // Control functions
  void Init(boolean invert);
  void GotoXY(uint8_t x, uint8_t y);
  // Graphic Functions
  void DrawLine(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t color);
  void DrawRect(uint8_t x, uint8_t y, uint8_t width, uint8_t height, uint8_t color);
  void DrawRoundRect(uint8_t x, uint8_t y, uint8_t width, uint8_t height, uint8_t radius, uint8_t color);
  void FillRect(uint8_t x, uint8_t y, uint8_t width, uint8_t height, uint8_t color);
  void InvertRect(uint8_t x, uint8_t y, uint8_t width, uint8_t height);
  void SetInverted(boolean invert);
  void SetDot(uint8_t x, uint8_t y, uint8_t color);
  void DrawBarGraph(uint8_t y, int8_t val);

  // Font Functions
  void SelectFont(const uint8_t* font, uint8_t color=BLACK, FontCallback callback=ReadFontData); // defualt arguments added, callback now last arg
  int PutChar(char c);
  void Puts(char* str);
  void PutsCentered(char* str, uint8_t x, uint8_t y, uint8_t width);
  void Puts_P(PGM_P str);
  void PrintNumber(long n);
  void PrintNumberCentered(long n, uint8_t x, uint8_t y, uint8_t width);

  uint8_t CharWidth(char c);
  uint16_t StringWidth(char* str);
  uint16_t StringWidth_P(PGM_P str);


 private:
  lcdCoord                        Coord;
  boolean                                Inverted;  // changed type to boolean
  FontCallback            FontRead;
  uint8_t                                FontColor;
  const uint8_t*                Font;
  uint8_t ReadData(void);  // TODO this was inline !!!
  uint8_t DoReadData(uint8_t first);
  void WriteData(uint8_t data);
  void WriteCommand(uint8_t cmd, uint8_t chip);
  inline void Enable(void);
};

extern ks0108iLab GLCDiLab;    
#endif
