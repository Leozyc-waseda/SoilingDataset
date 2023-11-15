'' ******************************************************************************
'' * MCP320016 i2c Bus Expander Object                                          *
'' * James Burrows May 2006                                                     *
'' * Version 1.1                                                                *
'' ******************************************************************************
''
'' MCP23018 i2c I/O expander object. Devied from MCP23016 Lior.
''
'' For a simple example write a 0 to the I/O register to set all 8 pins output, and
'' 255 or %1111_1111 to GP0 or GP1 to make the pins +5V.  Attach some LED's!  
''
'' www.microchip.com
''
'' this object provides the PUBLIC functions:
''  -> Init  - sets up the address and inits sub-objects such
''  -> WriteIOregister0 - set the output register bank 0
''  -> WriteIOregister1 - set the output register bank 1
''  -> WriteGP0 - set the GP register bank 0 
''  -> WriteGP1 - set the GP register bank 1
''  -> ReadGP0 - read the GP register bank 0
''  -> ReadGP1 - read the GP register bank 0
''
'' this object provides the PRIVATE functions:
''  -> None
''
'' this object uses the following sub OBJECTS:
''  -> i2cObject
''
'' Revision History:
''  -> V1    - Release
''  -> V1.1  - Updated to allow i2cSCL line driving pass-true to i2cObject
'' 
'' The default address is %0100_0000

CON
  ' MCP constants
  _MCP23016_IODIR0 = $00
  _MCP23016_IODIR1 = $01
  _MCP23016_GP0   = $12
  _MCP23016_GP1   = $13
  _MCP23016_OLATA   = $14
  _MCP23016_OLATB   = $15
  _MCP23016_GPPUA = $C
  _MCP23016_GPPUB = $D  


VAR
  long  MCP23016_Address
  long  started    


OBJ
  i2cObject     : "i2cObject"

  
pub Init(_deviceAddress,_i2cSDA,_i2cSCL,_driveSCLLine): okay
  ' init the Object
  MCP23016_Address := _deviceAddress
  i2cObject.init(_i2cSDA,_i2cSCL,_driveSCLLine)

  'start
  okay := start

  return okay

PUB start : okay
  ' try a restart
  if started == false
    if i2cObject.devicePresent(MCP23016_Address) == true
      started := true
    else
      started := false
  return started

PUB isStarted : result
  ' return the started state
  return started  
  
PUB WriteIOregister0(i2cData) : ackbit
  ' write to the I/O register port 0
  ' a 0==Output and a 1==Input
  if started == true
    ackbit := i2cObject.WriteLocation(MCP23016_Address, _MCP23016_IODIR0,i2cData,8,8)
    return ackbit  

PUB WriteIOregister1(i2cData) : ackbit
  ' write to the I/O register port 1
  ' a 0==Output and a 1==Input
  if started == true
    ackbit := i2cObject.WriteLocation(MCP23016_Address, _MCP23016_IODIR1,i2cData,8,8)
    return ackbit  

PUB ReadRegister(reg) : i2cData
  ' Read a Register
  if started == true
    i2cData := i2cObject.readLocation(MCP23016_Address, reg, 8, 8)
    return i2cData
PUB WriteRegister(reg, i2cData) : ackbit
  ' write to the I/O register port 1
  ' a 0==Output and a 1==Input
  if started == true
    ackbit := i2cObject.WriteLocation(MCP23016_Address, reg, i2cData,8,8)
    return ackbit  

PUB SetPullup0(i2cData) : ackbit
  if started == true
    ackbit := i2cObject.WriteLocation(MCP23016_Address, _MCP23016_GPPUA,i2cData,8,8)
    return ackbit  

PUB SetPullup1(i2cData) : ackbit
  if started == true
    ackbit := i2cObject.WriteLocation(MCP23016_Address, _MCP23016_GPPUB,i2cData,8,8)
    return ackbit  
     
PUB WriteGP0(i2cData) : ackbit
  ' write to the General Purpose (GP) Port register 0
  ' a 0==PinLow and 1==PinHigh
  if started == true 
    ackbit := i2cObject.WriteLocation(MCP23016_Address, _MCP23016_GP0,i2cData,8,8)
    return ackbit

PUB WriteOLAT0(i2cData) : ackbit
  ' write to the General Purpose (GP) Port register 0
  ' a 0==PinLow and 1==PinHigh
  if started == true 
    ackbit := i2cObject.WriteLocation(MCP23016_Address, _MCP23016_OLATA,i2cData,8,8)
    return ackbit

PUB WriteGP1(i2cData) : ackbit
  ' write to the General Purpose (GP) Port register 1
  ' a 0==PinLow and 1==PinHigh
  if started == true  
    ackbit := i2cObject.WriteLocation(MCP23016_Address, _MCP23016_GP1,i2cData,8,8)
    return ackbit

PUB ReadGP0 : i2cData
  ' Read the General Purpose (GP) Port register 0
  ' a 0==PinLow and 1==PinHigh
  if started == true
    i2cData := i2cObject.readLocation(MCP23016_Address, _MCP23016_GP0, 8, 8)
    return i2cData

PUB ReadOLAT0 : i2cData
  ' Read the General Purpose (GP) Port register 0
  ' a 0==PinLow and 1==PinHigh
  if started == true
    i2cData := i2cObject.readLocation(MCP23016_Address, _MCP23016_OLATA, 8, 8)
    return i2cData

PUB ReadGP1 : i2cData
  ' Read the General Purpose (GP) Port register 1
  ' a 0==PinLow and 1==PinHigh
  if started == true
    i2cData := i2cObject.readLocation(MCP23016_Address, _MCP23016_GP1, 8, 8)
    return i2cData
