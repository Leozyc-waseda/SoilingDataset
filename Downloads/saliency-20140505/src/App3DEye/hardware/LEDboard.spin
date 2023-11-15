CON
_clkmode = xtal1 + pll16x                           
_xinfreq = 5_000_000

pin1 = 3
pin2 = 4
pin3 = 5
pin4 = 6
pin5 = 25
pin6 = 26
pin7 = 27
pin8 = 28

VAR
long inByte
byte onVal[8]
long blinkSpd

OBJ
Ser     :       "FullDuplexSerial"                      ''Used in this DEMO for Debug

PUB start
Ser.start(31,30,0,115200)
dira[pin1]~~
dira[pin2]~~
dira[pin3]~~
dira[pin4]~~
dira[pin5]~~
dira[pin6]~~
dira[pin7]~~
dira[pin8]~~
outa := $FFFF_FFFF
waitcnt(clkfreq/5+cnt)
outa[pin1]~
waitcnt(clkfreq/5+cnt)
outa[pin1]~~
outa[pin2]~
waitcnt(clkfreq/5+cnt)
outa[pin2]~~
outa[pin3]~
waitcnt(clkfreq/5+cnt)
outa[pin3]~~
outa[pin4]~
waitcnt(clkfreq/5+cnt)
outa[pin4]~~
outa[pin5]~
waitcnt(clkfreq/5+cnt)
outa[pin5]~~
outa[pin6]~
waitcnt(clkfreq/5+cnt)
outa[pin6]~~
outa[pin7]~
waitcnt(clkfreq/5+cnt)
outa[pin7]~~
outa[pin8]~
waitcnt(clkfreq/5+cnt)
outa[pin8]~~
waitcnt(clkfreq/5+cnt)
outa[pin1]~
outa[pin2]~
outa[pin3]~
outa[pin4]~
outa[pin5]~
outa[pin6]~
outa[pin7]~
outa[pin8]~
waitcnt(clkfreq+cnt)
onVal[0] := -1
onVal[1] := -1
onVal[2] := -1
onVal[3] := -1
onVal[4] := -1
onVal[5] := -1
onVal[6] := -1
onVal[7] := -1
blinkSpd := 50
repeat
  ''Ser.str(string("Test "))
  inByte := Ser.rxcheck
  if (inByte <> -1)
    'Ser.dec(inByte)
    if (inByte == 0)
      Ser.str(string("ledboard"))
    elseif(inByte == 8)
      onVal[0] := 0
      onVal[1] := 0
      onVal[2] := 0
      onVal[3] := 0
      onVal[4] := 0
      onVal[5] := 0
      onVal[6] := 0
      onVal[7] := 0      
    elseif(inByte == 13)
      onVal[0] := -1
      onVal[1] := -1
      onVal[2] := -1
      onVal[3] := -1
      onVal[4] := -1
      onVal[5] := -1
      onVal[6] := -1
      onVal[7] := -1      
    elseif(inByte < 100)
      if(inByte == 49)
        onVal[0] := !onVal[0]
      elseif(inByte == 50)
        onVal[1] := !onVal[1]
      elseif(inByte == 51)
        onVal[2] := !onVal[2]       
      elseif(inByte == 52)
        onVal[3] := !onVal[3]      
      elseif(inByte == 53)
        onVal[4] := !onVal[4]
      elseif(inByte == 54)
        onVal[5] := !onVal[5]
      elseif(inByte == 55)
        onVal[6] := !onVal[6]
      elseif(inByte == 56)
        onVal[7] := !onVal[7]
    elseif(inByte < 110)
      if(inByte == 101)
        onVal[0]~~
      elseif(inByte == 102)
        onVal[1]~~
      elseif(inByte == 103)
        onVal[2]~~
      elseif(inByte == 104)
        onVal[3]~~
      elseif(inByte == 105)
        onVal[4]~~
      elseif(inByte == 106)
        onVal[5]~~
      elseif(inByte == 107)
        onVal[6]~~
      elseif(inByte == 108)
        onVal[7]~~

    elseif(inByte > 110)
      if(inByte == 111)
        onVal[0]~
      elseif(inByte == 112)
        onVal[1]~
      elseif(inByte == 113)
        onVal[2]~
      elseif(inByte == 114)
        onVal[3]~
      elseif(inByte == 115)
        onVal[4]~
      elseif(inByte == 116)
        onVal[5]~
      elseif(inByte == 117)
        onVal[6]~
      elseif(inByte == 118)
        onVal[7]~
      elseif(inByte == 200)
        inByte := Ser.rxtime(50)
        if inByte <> -1
          blinkSpd := inByte * 10
        
      
    'Ser.rxflush
  'outa[pin1]~  
  'outa[pin2]~
  'outa[pin3]~
  'outa[pin4]~
  'outa[pin5]~
  'outa[pin6]~
  'outa[pin7]~
  'outa[pin8]~
  'outa := $00000000
  outa[pin1] := onVal[0]
  outa[pin2] := onVal[1]
  outa[pin3] := onVal[2]
  outa[pin4] := onVal[3]
  outa[pin5] := onVal[4]
  outa[pin6] := onVal[5]
  outa[pin7] := onVal[6]
  outa[pin8] := onVal[7]    
  waitcnt(clkfreq/1000*blinkSpd+cnt)
  outa := $FFFFFFFF
  waitcnt(clkfreq/1000*blinkSpd+cnt)