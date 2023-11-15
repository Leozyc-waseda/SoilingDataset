
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

  Version:   1.0 - May 8 2008
  
*/
#include "ks0108iLab.h"
extern "C" {
#include <inttypes.h>
#include <avr/io.h>
#include <avr/pgmspace.h>
#include <wiring.h> // added 18 Sept 2008 for Arduino release 0012

}
//#define GLCD_DEBUG  // uncomment this if you want to slow down drawing to see how pixels are set

// ######################################################################
void ks0108iLab::DrawLine(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t color) {
  uint8_t length, i, y, yAlt, xTmp, yTmp;
  int16_t m;
  //
  // vertical line
  //
  if (x1 == x2) {
    // x1|y1 must be the upper point
    if (y1 > y2) this->DrawVertLine(x1, y2, y1-y2+1, color);
    else this->DrawVertLine(x1, y1, y2-y1+1, color);
    //
    // horizontal line
    //
  } else if (y1 == y2) {        
    // x1|y1 must be the left point
    if (x1 > x2) this->DrawHoriLine(x2, y1, x1-x2+1, color);
    else this->DrawHoriLine(x1, y1, x2-x1+1, color);
    //
    // schiefe line :)
    //
  } else {
    // angle >= 45deg
    if ((y2-y1) >= (x2-x1) || (y1-y2) >= (x2-x1)) {
      // x1 must be smaller than x2
      if (x1 > x2) { xTmp = x1; yTmp = y1; x1 = x2; y1 = y2; x2 = xTmp; y2 = yTmp; }
      
      length = x2-x1;                // not really the length :)
      m = ((y2-y1)*200) / length;
      yAlt = y1;
      
      for (i = 0; i <= length; ++i) {
        y = ((m*i)/200)+y1;
        
        if ((m*i) % 200 >= 100) ++y;
        else if ((m*i) % 200 <= -100) --y;
        
        this->DrawLine(x1+i, yAlt, x1+i, y, color);
        
        if (length <= (y2-y1) && y1 < y2) yAlt = y+1;
        else if (length <= (y1-y2) && y1 > y2) yAlt = y-1;
        else yAlt = y;
      }
      // angle < 45deg
    } else {
      // y1 must be smaller than y2
      if (y1 > y2) { xTmp = x1; yTmp = y1; x1 = x2; y1 = y2; x2 = xTmp; y2 = yTmp; }
      
      length = y2 - y1;
      m = ((x2-x1)*200) / length;
      yAlt = x1;
      
      for (i = 0; i <= length; ++i) {
        y = ((m*i)/200)+x1;
        
        if ((m*i) % 200 >= 100) ++y;
        else if((m*i) % 200 <= -100) --y;
        
        this->DrawLine(yAlt, y1+i, y, y1+i, color);
        if (length <= (x2-x1) && x1 < x2) yAlt = y+1;
        else if (length <= (x1-x2) && x1 > x2) yAlt = y-1;
        else yAlt = y;
      }
    }
  }
}

// ######################################################################
void ks0108iLab::DrawRect(uint8_t x, uint8_t y, uint8_t width, uint8_t height, uint8_t color) {
  DrawHoriLine(x, y, width, color);            // top
  DrawHoriLine(x, y+height-1, width, color);   // bottom
  DrawVertLine(x, y, height, color);           // left
  DrawVertLine(x+width-1, y, height, color);   // right
}

// ######################################################################
void ks0108iLab::DrawRoundRect(uint8_t x, uint8_t y, uint8_t width, uint8_t height,
                           uint8_t radius, uint8_t color) {
  int16_t tSwitch, x1 = 0, y1 = radius;
  tSwitch = 3 - 2 * radius;
  
  while (x1 <= y1) {
    this->SetDot(x+radius - x1, y+radius - y1, color);
    this->SetDot(x+radius - y1, y+radius - x1, color);
    
    this->SetDot(x+width-1-radius + x1, y+radius - y1, color);
    this->SetDot(x+width-1-radius + y1, y+radius - x1, color);
    
    this->SetDot(x+width-1-radius + x1, y+height-1-radius + y1, color);
    this->SetDot(x+width-1-radius + y1, y+height-1-radius + x1, color);
    
    this->SetDot(x+radius - x1, y+height-1-radius + y1, color);
    this->SetDot(x+radius - y1, y+height-1-radius + x1, color);
    
    if (tSwitch < 0) tSwitch += (4 * x1 + 6);
    else { tSwitch += (4 * (x1 - y1) + 10); --y1; }
    ++x1;
  }
  
  this->DrawHoriLine(x+radius, y, width-(2*radius), color);           // top
  this->DrawHoriLine(x+radius, y+height-1, width-(2*radius), color);  // bottom
  this->DrawVertLine(x, y+radius, height-(2*radius), color);          // left
  this->DrawVertLine(x+width-1, y+radius, height-(2*radius), color);  // right
}

/*
 * Hardware-Functions 
 */
// ######################################################################
void ks0108iLab::FillRect(uint8_t x, uint8_t y, uint8_t width, uint8_t height, uint8_t color) {
  uint8_t h, i, data;
  uint8_t pageOffset = y % 8; // offset of top of rect wrt 8-pixel vertical pages
  y -= pageOffset;            // y rounded to lower 8-pixel page boundary

  // top rows: go from pageOffset down, either to the whole byte or
  // less if the height is smaller than pageOffset:
  uint8_t mask = 0xFF;
  if (height < 8 - pageOffset) { mask >>= (8-height); h = height; } else h = 8-pageOffset;
  mask <<= pageOffset;
        
  this->GotoXY(x, y);
  if (color == BLACK)
    for (i = 0; i < width; ++i) { data = this->ReadData(); data |= mask; this->WriteData(data); }
  else {
    mask = ~mask;
    for (i = 0; i < width; ++i) { data = this->ReadData(); data &= mask; this->WriteData(data); }
  }

  // bulk of the rectangle
  while (h+8 < height)
    { h += 8; y += 8; this->GotoXY(x, y); for (i = 0; i < width; ++i) this->WriteData(color); }

  // bottom rows (if bottom of rect on on an 8-row boundary)
  if (h < height) {
    mask = ~(0xFF << (height-h));
    this->GotoXY(x, y+8);

    if (color == BLACK)
      for (i = 0; i < width; ++i) { data = this->ReadData(); data |= mask; this->WriteData(data); }
    else {
      mask = ~mask;
      for (i = 0; i < width; ++i) { data = this->ReadData(); data &= mask; this->WriteData(data); }
    }
  }
}

// ######################################################################
void ks0108iLab::InvertRect(uint8_t x, uint8_t y, uint8_t width, uint8_t height) {
  uint8_t mask, pageOffset, h, i, data, tmpData;
        
  pageOffset = y%8;
  y -= pageOffset;
  mask = 0xFF;
  if(height < 8-pageOffset) {
    mask >>= (8-height);
    h = height;
  } else {
    h = 8-pageOffset;
  }
  mask <<= pageOffset;
        
  this->GotoXY(x, y);
  for (i = 0; i < width; ++i) {
    data = this->ReadData();
    tmpData = ~data;
    data = (tmpData & mask) | (data & ~mask);
    this->WriteData(data);
  }
  
  while (h+8 < height) {
    h += 8;
    y += 8;
    this->GotoXY(x, y);
    
    for (i = 0; i < width; ++i) {
      data = this->ReadData();
      this->WriteData(~data);
    }
  }
        
  if(h < height) {
    mask = ~(0xFF << (height-h));
    this->GotoXY(x, y+8);
                
    for (i = 0; i < width; ++i) {
      data = this->ReadData();
      tmpData = ~data;
      data = (tmpData & mask) | (data & ~mask);
      this->WriteData(data);
    }
  }
}

// ######################################################################
void ks0108iLab::SetInverted(boolean invert) {  // changed type to boolean
  if (this->Inverted != invert) {
    this->InvertRect(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT);
    this->Inverted = invert;
  }
}

// ######################################################################
void ks0108iLab::SetDot(uint8_t x, uint8_t y, uint8_t color) {
  this->GotoXY(x, y - y % 8);
  uint8_t data = this->ReadData();
  if (color == BLACK) data |= (0x01 << (y % 8)); else data &= ~(0x01 << (y % 8));
  this->WriteData(data);
}

// ######################################################################
void ks0108iLab::DrawBarGraph(uint8_t y, int8_t val) {
  uint8_t xx = 62, yy = y;
  uint8_t ww = 4, hh = 4;  // bar width and height
  int8_t step = ww + 1; // x step

  if (val >= 0)
    {
      if (val > 60) val = 60;

      // draw the bars proper:
      int8_t i;
      for (i = 0; i <= val; i += step) {
        this->FillRect(xx, yy, ww, hh, BLACK);
        xx += step; yy -= 1; hh += 2;
      }
      // clear bars that correspond to higher values than val:
      for (; i <= 60; i += step) {
        this->FillRect(xx, yy, ww, hh, WHITE);
        xx += step; yy -= 1; hh += 2;
      }
      // clear all bars on the negative side:
      xx = 62 - step; yy = y - 1; hh = 6;
      for (i = -1; i >= -60; i -= step) {
        this->FillRect(xx, yy, ww, hh, WHITE);
        xx -= step; yy -= 1; hh += 2;
      }
    }
  else
    {
      if (val < -60) val = -60;

      // draw the bars proper:
      int8_t i;
      for (i = 0; i >= val; i -= step) {
        this->FillRect(xx, yy, ww, hh, BLACK);
        xx -= step; yy -= 1; hh += 2;
      }
      // clear bars that correspond to lower values than val:
      for (i = 0; i >= -60; i -= step) {
        this->FillRect(xx, yy, ww, hh, WHITE);
        xx -= step; yy -= 1; hh += 2;
      }
      // clear all bars on the positive side:
      xx = 62 + step; yy = y - 1; hh = 6;
      for (i = 1; i <= 60; i += step) {
        this->FillRect(xx, yy, ww, hh, WHITE);
        xx += step; yy -= 1; hh += 2;
      }
    }
}

//
// Font Functions
//

// ######################################################################
uint8_t ReadFontData(const uint8_t* ptr) {  // note this is a static function
  return pgm_read_byte(ptr);
}

// ######################################################################
void ks0108iLab::SelectFont(const uint8_t* font,uint8_t color, FontCallback callback) {
  this->Font = font;
  this->FontRead = callback;
  this->FontColor = color;
}

// ######################################################################
void ks0108iLab::PrintNumber(long n) {
  byte buf[10];  // prints up to 10 digits (max for a long)
  byte *bptr = buf;
  if (n == 0) PutChar('0');
  else {
    if (n < 0) { PutChar('-'); n = -n; }
    while (n > 0) { *bptr++ = n % 10; n /= 10; }
    for (--bptr; bptr >= buf; --bptr) this->PutChar((char)('0' + *bptr));
  }
}

// ######################################################################
void ks0108iLab::PrintNumberCentered(long n, uint8_t x, uint8_t y, uint8_t width) {
  char str[12];  // prints up to 10 digits (max for a long) + sign + null
  char *sptr = &(str[11]);
  *sptr = 0; // null string terminator

  if (n == 0) *(--sptr) = '0';
  else {
    long nn = n;
    if (n < 0) n = -n;
    while (n > 0) { *(--sptr) = (n % 10) + '0'; n /= 10; }
    if (nn < 0) *(--sptr) = '-';
  }

  this->PutsCentered(sptr, x, y, width);
}

// ######################################################################
int ks0108iLab::PutChar(char c) {
  uint8_t width = 0;
  uint8_t height = this->FontRead(this->Font + FONT_HEIGHT);
  uint8_t bytes = (height + 7) / 8;
        
  uint8_t firstChar = this->FontRead(this->Font + FONT_FIRST_CHAR);
  uint8_t charCount = this->FontRead(this->Font + FONT_CHAR_COUNT);
        
  uint16_t index = 0;
  uint8_t x = this->Coord.x, y = this->Coord.y;

  if (c < firstChar || c >= (firstChar+charCount)) return 1;
  c -= firstChar;
        
  // read width data, to get the index
  for (uint8_t i = 0; i < c; ++i)
    index += this->FontRead(this->Font + FONT_WIDTH_TABLE + i);

  index = index * bytes + charCount + FONT_WIDTH_TABLE;
  width = this->FontRead(this->Font + FONT_WIDTH_TABLE + c);
        
  // last but not least, draw the character
  for (uint8_t i = 0; i < bytes; ++i) {
    uint8_t page = i * width;
    for (uint8_t j = 0; j < width; ++j) {
      uint8_t data = this->FontRead(this->Font + index + page + j);
      if (height < (i+1)*8) data >>= (i+1)*8 - height;
      if (this->FontColor == BLACK) this->WriteData(data); else this->WriteData(~data);
    }

    // 1px gap between chars
    if (this->FontColor == BLACK) this->WriteData(0x00); else this->WriteData(0xFF);
    this->GotoXY(x, this->Coord.y + 8);
  }
  this->GotoXY(x + width + 1, y);

  return 0;
}

// ######################################################################
void ks0108iLab::Puts(char* str) {
  int x = this->Coord.x;
  while (*str != 0) {
    if (*str == '\n') this->GotoXY(x, this->Coord.y+this->FontRead(this->Font+FONT_HEIGHT));
    else this->PutChar(*str);
    ++str;
  }
}

// ######################################################################
void ks0108iLab::PutsCentered(char *str, uint8_t x, uint8_t y, uint8_t width) {
  uint16_t w = this->StringWidth(str);

  // is the string is too long (or exactly)? if so, just Puts, it may overflow
  if (w >= (uint16_t)(width)) { this->GotoXY(x, y); this->Puts(str); return; }

  // at this point, we know that w < width (and < 256)
  uint8_t ww = (uint8_t)(w);
  uint8_t ww1 = (width - ww) / 2;
  uint8_t ww2 = ww1;
  if ((width - ww) & 1) ++ww2; // make sure ww1 + w + ww2 = width
  uint8_t h = this->FontRead(this->Font + FONT_HEIGHT);

  if (ww1) this->FillRect(x, y, ww1, h, ~(this->FontColor));
  this->GotoXY(x + ww1, y); this->Puts(str);
  if (ww2) this->FillRect(x + ww1 + ww, y, ww2, h, ~(this->FontColor));
}

// ######################################################################
void ks0108iLab::Puts_P(PGM_P str) {
  int x = this->Coord.x;
  while (pgm_read_byte(str) != 0) {
    if (pgm_read_byte(str) == '\n') this->GotoXY(x, this->Coord.y+this->FontRead(this->Font+FONT_HEIGHT));
    else this->PutChar(pgm_read_byte(str));
    ++str;
  }
}

// ######################################################################
uint8_t ks0108iLab::CharWidth(char c) {
  uint8_t width = 0;
  uint8_t firstChar = this->FontRead(this->Font+FONT_FIRST_CHAR);
  uint8_t charCount = this->FontRead(this->Font+FONT_CHAR_COUNT);

  // read width data
  if (c >= firstChar && c < (firstChar+charCount)) {
    c -= firstChar;
    width = this->FontRead(this->Font+FONT_WIDTH_TABLE+c)+1;
  }

  return width;
}

// ######################################################################
uint16_t ks0108iLab::StringWidth(char* str) {
  uint16_t width = 0;
  while (*str != 0) width += this->CharWidth(*str++);
  return width;
}

// ######################################################################
uint16_t ks0108iLab::StringWidth_P(PGM_P str) {
  uint16_t width = 0;
  while(pgm_read_byte(str) != 0) width += this->CharWidth(pgm_read_byte(str++));
  return width;
}

// ######################################################################
void ks0108iLab::GotoXY(uint8_t x, uint8_t y) {
  uint8_t chip = CHIP1, cmd;
  
  if (x >= DISPLAY_WIDTH) x = DISPLAY_WIDTH-1;                 // ensure that coordinates are legal
  if (y >= DISPLAY_HEIGHT) y = DISPLAY_HEIGHT-1;
  
  this->Coord.x = x;                                                                // save new coordinates
  this->Coord.y = y;
  this->Coord.page = y/8;
  
  if(x >= CHIP_WIDTH) {                                                        // select the right chip
    x -= CHIP_WIDTH;
    chip = CHIP2;
  }
  
#ifdef HD44102 
  this->WriteCommand(LCD_DISP_START, CHIP1);                // display start line = 0
  this->WriteCommand(LCD_DISP_START, CHIP2);
  
  cmd = (this->Coord.page << 6 ) | x;
  //        this->WriteCommand(cmd,chip);
  this->WriteCommand(cmd, CHIP1);                
  this->WriteCommand(cmd, CHIP2);
#else
  cmd = LCD_SET_ADD | x;
  this->WriteCommand(cmd, chip);                                        // set x address on active chip
  
  cmd = LCD_SET_PAGE | this->Coord.page;                        // set y address on both chips
  this->WriteCommand(cmd, CHIP1);
  this->WriteCommand(cmd, CHIP2);
#endif
}

// ######################################################################
void ks0108iLab::Init(boolean invert) {  
  pinMode(D_I,OUTPUT);         
  pinMode(R_W,OUTPUT);         
  pinMode(EN,OUTPUT);         
  pinMode(CSEL1,OUTPUT);
  pinMode(CSEL2,OUTPUT);
  pinMode(13,OUTPUT); // for testing only !!!
  
  this->Coord.x = 0;
  this->Coord.y = 0;
  this->Coord.page = 0;
        
  this->Inverted = invert;
  
  this->WriteCommand(LCD_ON, CHIP1);                                // power on
  this->WriteCommand(LCD_ON, CHIP2);
  
  this->WriteCommand(LCD_DISP_START, CHIP1);                // display start line = 0
  this->WriteCommand(LCD_DISP_START, CHIP2);
  
  this->ClearScreen();                                                        // display clear
  this->GotoXY(0,0);
}

// ######################################################################
static void delay450ns(void){   // delay 450 nanoseconds
  asm volatile("nop\n\t" 
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"  // todo, remove some nops if clock is less than 16mhz
               "nop\n\t"
               
               // LI: default was 5 nops, seems too fast for sparkfun lcd
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t" // still too fast
               
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t" // seems ok
               
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t"
               "nop\n\t" // just to be sure

               ::);
}

 // ######################################################################
__inline__ void ks0108iLab::Enable(void) {  
  fastWriteHigh(EN);                                        // EN high level width: min. 450ns
  delay450ns();
  fastWriteLow(EN);
  // for(volatile uint8_t i=0; i<8; i++);  // big delay loop in Fabian's code, was 5.7us, replaced by call to 450ns delay function
  delay450ns();
}

// ######################################################################
uint8_t ks0108iLab::DoReadData(uint8_t first) {
  uint8_t data;
  
  lcdDataOut(0x00);
  lcdDataDir(0x00);                                        // data port is input
  
  if(this->Coord.x < CHIP_WIDTH) {
    fastWriteLow(CSEL2);                        // deselect chip 2
    fastWriteHigh(CSEL1);                        // select chip 1
  } else if(this->Coord.x >= CHIP_WIDTH) {
    fastWriteLow(CSEL1);                        // deselect chip 1
    fastWriteHigh(CSEL2);                        // select chip 2
  }
  if(this->Coord.x == CHIP_WIDTH && first) {                // chip2 X-address = 0
    this->WriteCommand(LCD_SET_ADD, CHIP2);         // wuff wuff
  }
  
  fastWriteHigh(D_I);                                        // D/I = 1
  fastWriteHigh(R_W);                                        // R/W = 1
  
  fastWriteHigh(EN);                                         // EN high level width: min. 450ns
  delay450ns();
#ifdef LCD_DATA_NIBBLES
  data = (LCD_DATA_IN_LOW & 0x0F) | (LCD_DATA_IN_HIGH & 0xF0);
#else
  data = LCD_DATA_IN_LOW;                           // low and high nibbles on same port so read all 8 bits at once
#endif 
  
  fastWriteLow(EN); 
  // for(volatile uint8_t i=0; i<8; i++);  // big delay loop in Fabian's code, was 5.7us, replaced by 450ns delay below
  delay450ns();
  
  lcdDataDir(0xFF);
  
  this->GotoXY(this->Coord.x, this->Coord.y);
  
  if(this->Inverted)
    data = ~data;
  return data;
}

// ######################################################################
inline uint8_t ks0108iLab::ReadData(void) {  
  this->DoReadData(1);                                // dummy read
  return this->DoReadData(0);                        // "real" read
}

// ######################################################################
void ks0108iLab::WriteCommand(uint8_t cmd, uint8_t chip) {
  if(chip == CHIP1) {
    fastWriteLow(CSEL2);                        // deselect chip 2
    fastWriteHigh(CSEL1);                        // select chip 1
  } else if(chip == CHIP2) {
    fastWriteLow(CSEL1);                        // deselect chip 1
    fastWriteHigh(CSEL2);                        // select chip 2
  }
  fastWriteLow(D_I);                                        // D/I = 0
  fastWriteLow(R_W);                                        // R/W = 0        
  
  lcdDataDir(0xFF);
  lcdDataOut(cmd);
  this->Enable();                                                // enable
  lcdDataOut(0x00);
}

// ######################################################################
void ks0108iLab::WriteData(uint8_t data) {
  uint8_t displayData, yOffset;
#ifdef LCD_CMD_PORT        
  uint8_t cmdPort;        
#endif
  
#ifdef GLCD_DEBUG
  volatile uint16_t i;
  for(i=0; i<5000; i++);
#endif
  
  if(this->Coord.x >= DISPLAY_WIDTH)
    return;
  
  if(this->Coord.x < CHIP_WIDTH) {
    fastWriteLow(CSEL2);                        // deselect chip 2
    fastWriteHigh(CSEL1);                        // select chip 1
  } else {
    fastWriteLow(CSEL1);                        // deselect chip 1
    fastWriteHigh(CSEL2);                        // select chip 2
  }
#ifndef HD44102
  if(this->Coord.x == CHIP_WIDTH)                                                        // chip2 X-address = 0
    this->WriteCommand(LCD_SET_ADD, CHIP2);
#endif        
  fastWriteHigh(D_I);                                        // D/I = 1
  fastWriteLow(R_W);                                  // R/W = 0        
  lcdDataDir(0xFF);                                        // data port is output
  
  
  yOffset = this->Coord.y%8;
  if(yOffset != 0) {
    // first page
#ifdef LCD_CMD_PORT 
    cmdPort = LCD_CMD_PORT;                                                // save command port
#endif
    displayData = this->ReadData();
#ifdef LCD_CMD_PORT                 
    LCD_CMD_PORT = cmdPort;                                                // restore command port
#else
    fastWriteHigh(D_I);                                        // D/I = 1
    fastWriteLow(R_W);                                        // R/W = 0
    if(this->Coord.x < CHIP_WIDTH) {
      fastWriteLow(CSEL2);                        // deselect chip 2
      fastWriteHigh(CSEL1);                        // select chip 1
    } else {
      fastWriteLow(CSEL1);                        // deselect chip 1
      fastWriteHigh(CSEL2);                        // select chip 2
    }
#endif
    lcdDataDir(0xFF);                                                // data port is output
    
    displayData |= data << yOffset;
    if(this->Inverted)
      displayData = ~displayData;
    lcdDataOut( displayData);                                        // write data
    this->Enable();                                                                // enable
    
    // second page
    this->GotoXY(this->Coord.x, this->Coord.y+8);
    
    displayData = this->ReadData();
    
#ifdef LCD_CMD_PORT                 
    LCD_CMD_PORT = cmdPort;                                                // restore command port
#else                
    fastWriteHigh(D_I);                                        // D/I = 1
    fastWriteLow(R_W);                                         // R/W = 0        
    if(this->Coord.x < CHIP_WIDTH) {
      fastWriteLow(CSEL2);                        // deselect chip 2
      fastWriteHigh(CSEL1);                        // select chip 1
    } else {
      fastWriteLow(CSEL1);                        // deselect chip 1
      fastWriteHigh(CSEL2);                        // select chip 2
    }
#endif
    lcdDataDir(0xFF);                                // data port is output
    
    displayData |= data >> (8-yOffset);
    if(this->Inverted)
      displayData = ~displayData;
    lcdDataOut(displayData);                // write data
    this->Enable();                                        // enable
    
    this->GotoXY(this->Coord.x+1, this->Coord.y-8);
  } else {
    if(this->Inverted)
      data = ~data;
    lcdDataOut(data);                                // write data
    this->Enable();                                        // enable
    this->Coord.x++;
  }
  lcdDataOut(0x00);
}

// ######################################################################
// class wrapper
ks0108iLab::ks0108iLab(){
  this->Inverted=0;
}

// ######################################################################
// Make one instance for the user
ks0108iLab GLCDiLab = ks0108iLab();
