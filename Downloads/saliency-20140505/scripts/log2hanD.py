#! /usr/bin/env python

# >>> Comments <<<
# Function: Convert windows' joystick movement log into a separate file called
#           .hanD and possibly some extra files
# Expected extension:
# .log  >>> the windows' joystick file
# .psy  >>> the psycho file that contains master time recording in linux box
# .hanD >>> expected output. contains only bunch of numbers

### Please do version control here (manually)
#
# v4.0 >>> Added keyboard and mouse
#
# v3.0 >>> Added sync function in log file
#
# v1.2 >>> Added more option to slow compensator
#
# v1.1 >>> Added the dupes & slow frames detection & simple fix
#
# v1.0 >>> Is able to parse .log and .psy into .hanD plus some minor stuffs,
# still have some TODOs
#

### Some libraries here
import sys
#import string
import datetime

### Some constant values
itsRateExt  = ".rate"
itsTrashExt = ".ntrash"
itsOffsetExt= ".offset"
itsTempExt  = ".tmp"
enExtraFiles = False
enSlow = True
enLogTime = False
joyEn = False   ## This will detect whether need to put keyboard or not
keyEn = False
mouEn = False
joyButNo = 0 ## number of detected buttons

################# Functions #################

def cleanFile(theFilename): #TODO: overwrite protection here
    ff = open(theFilename, 'w')
    ff.close()
    return

def writeFile(theFilename, theText):
    ff = open(theFilename, 'a')
    ff.write(theText + "\n")
    ff.close()
    return

def itsShortUsage():
    print "Usage: ./Logger.py --log=<filename> [--psy=<filename>] [--target=<filename>]"
    print "       [--enable-extra] [--enable-logtime] [--res=<width>x<height>]"
    print "       [--compensation=<framelag>]"
    print ""
    return

def itsUsage():
    print "###################################"
    print "Usage: ./Logger.py --log=<filename> [--psy=<filename>] [--target=<filename>]"
    print "       [--enable-extra] [--enable-logtime] [--res=<width>x<height>]"
    print "       [--compensation=<framelag>]"
    print "###################################"
    print "Definition:"
    print "--log    : source file. it should contain the joystick/mouse/keyboard input log"
    print "--psy    : psy file. get from psycho-video, default: same as input w/ .psy ext"
    print "--target : target file. default: same as input with .hanD extension"
    print "--enable-extra  : remove the extra files : .rate & .ntrash, default: disabled"
    print "                  this will integrate the above parameters into the .hanD file"
    print "--enable-logtime: use the time from log and scretch it or make the"
    print "                  time to be constant. default: disabled/not using"
    print "--res    : screen resolution, default: [640x480]"
    print "--compensation : lag of time in frame"
    print ""
    print ""
    print ""
    return
    
def rescaleTime (baseTime, keyTime, actTime, scaleFactor):
    tmp = actTime - keyTime#baseTime
    tmp2 = tmp.microseconds + (tmp.seconds * 1000000)
    tmp3 = int(float(tmp2) * scaleFactor)
    tmp4 = datetime.timedelta(seconds=(tmp3/1000000),
                              microseconds=(tmp3%1000000))
    tmp = baseTime + tmp4
    #print str(tmp) +"  "+ str(tmp4) +"  "+ str(actTime)
    return tmp


################# Main Program #################

### To tell user how to use this
if (len(sys.argv) < 2 or len(sys.argv) > 6):
    itsUsage()
    exit(0)

### Parse the arguments

#defaults:
itsFilename   = ""
itsPsyname    = ""
itsTargetname = ""
itsOffset = 0
itsTrash  = 0
itsScreenSize = "640x480"
### compensation for lag, this means lag in frame
# It seems that our logger has about 2 frame delay
itsCompensation = 2


for itsArg in sys.argv[1:]:
    temp1 = itsArg.split('=')
    if (temp1[0] == "--log"):
        itsFilename = temp1[1]
    elif (temp1[0] == "--psy"):
        itsPsyname = temp1[1]
    elif (temp1[0] == "--target"):
        itsTargetname = temp1[1]
    elif (temp1[0] == "--enable-extra"):
        enExtraFiles = True
    elif (temp1[0] == "--enable-logtime"):
        enLogTime = True
    elif (temp1[0] == "--res"):
        itsScreenSize = temp1[1]
    elif (temp1[0] == "--compensation"):
        itsCompensation = atoi(temp1[1])
    else:
        print "Unknown argument : " + temp1[0]
        itsShortUsage()
        exit(0)

### Check whether they are still in default or not, especially itsFilename
if (itsFilename == ""):
    print "ERROR: source file input is not defined"
    itsShortUsage()
    exit(0)
    
#Make base path and filename
tmp2 = itsFilename.rsplit('/',1)
if (len(tmp2) == 1):
    itsBasename = tmp2[0].rsplit('.',1)[0]
    itsBasefolder = ""
else:
    itsBasename = tmp2[1].rsplit('.',1)[0]
    itsBasefolder = tmp2[0] + "/"
#default for psy file
if (itsPsyname == ""):
    itsPsyname = itsBasefolder + itsBasename + ".psy"
#default for target file
if (itsTargetname == ""):
    itsTargetname = itsBasefolder + itsBasename + ".hanD"

### Check targetname extension, add .hanD if its not there yet
if (not itsTargetname.endswith(".hanD")):
    itsTargetname += ".hanD"

### Additional files aside the .hanD file
itsRatename  = itsTargetname + itsRateExt
itsTrashname = itsTargetname + itsTrashExt
itsOffsetname= itsTargetname + itsOffsetExt
itsTempname  = itsTargetname + itsTempExt

### Clean the file up, TODO: add overwrite check
cleanFile(itsTargetname)

### Start computing everything <----------------------------------------------------------------------------------------
#Print all the parameter used
print "Computing using the following parameters"
print "Source      = "+itsFilename
print "Psy file    = "+itsPsyname
print "Target      = "+itsTargetname
print "Offset Time = "+str(itsOffset)
print "Extra data  = "+str(enExtraFiles)
print "Logtimeusage= "+str(enLogTime)

### Parsing the psy file if available
recordTime = datetime.timedelta.min
stopTime = datetime.timedelta()
psyAvail = True
try:
    print "### Start parsing the psy file ###"
    f_psy = open(itsPsyname, 'r')
except IOError:
    print "Cannot open " + itsPsyname + " file, ignoring the record time"
    psyAvail = False
else:
    for line in f_psy:
        temp_psy = line.split()
        # doing nested if instead of (A && B) to prevent out of range index error
        if (temp_psy[1] == "=="):
            if (temp_psy[3] == "RECORDING:"):
                #starting time
                t_r1 = temp_psy[4].split(':') # [0] = hr,  [1] = min
                t_r2 = t_r1[2].split('.')     # [0] = sec, [1] = ms, [2] = us
                recordTime = datetime.timedelta(hours=int(t_r1[0]),
                                                minutes=int(t_r1[1]),
                                                seconds=int(t_r2[0]),
                                                milliseconds=int(t_r2[1]),
                                                microseconds=int(t_r2[2]))
                #timestamp
                t_s1 = temp_psy[0].split(':') # [0] = hr,  [1] = min
                t_s2 = t_s1[2].split('.')     # [0] = sec, [1] = ms, [2] = us
                t1 = datetime.timedelta(hours=int(t_s1[0]),
                                        minutes=int(t_s1[1]),
                                        seconds=int(t_s2[0]),
                                        milliseconds=int(t_s2[1]),
                                        microseconds=int(t_s2[2]))
        elif (temp_psy[1] == "displayYUVoverlay"):
            if (temp_psy[4] == "0"): # The very first frame
                #timestamp
                t_s1 = temp_psy[0].split(':') # [0] = hr,  [1] = min
                t_s2 = t_s1[2].split('.')     # [0] = sec, [1] = ms, [2] = us
                t2 = datetime.timedelta(hours=int(t_s1[0]),
                                        minutes=int(t_s1[1]),
                                        seconds=int(t_s2[0]),
                                        milliseconds=int(t_s2[1]),
                                        microseconds=int(t_s2[2]))
        elif (temp_psy[1] == "-----"):
            if (temp_psy[4] == "Stop"):
                if (temp_psy[6] == "9"): #Stop @ Session #9, the real game
                    #timestamp
                    t_s1 = temp_psy[0].split(':')# [0] = hr,  [1] = min
                    t_s2 = t_s1[2].split('.')    # [0] = sec, [1] = ms, [2] = us
                    t3 = datetime.timedelta(hours=int(t_s1[0]),
                                            minutes=int(t_s1[1]),
                                            seconds=int(t_s2[0]),
                                            milliseconds=int(t_s2[1]),
                                            microseconds=int(t_s2[2]))
    f_psy.close()
    # Calculate the time difference between starting and stopping. Includes also
    # the offset
    dt = t2 - t1
    recordTime += dt + datetime.timedelta(milliseconds=itsOffset)
    dt = t3 - t2
    stopTime = recordTime + dt

    diffRecStopTime = stopTime - recordTime
    itsTotalTime = (diffRecStopTime.seconds * 1.0) + (diffRecStopTime.microseconds / 1000000.0)
    
    print "Start recording data : "+str(recordTime)
    print "Stop  recording data : "+str(stopTime)
    print "Total Recording time = "+str(diffRecStopTime.seconds + diffRecStopTime.microseconds/1000000.0) + " seconds = " + str(diffRecStopTime)


### Redo the reading again
### This time we are sure that whether the psy file is present or not
### we want the keyframe (frame xxxx99th frame) to be calculated
timeKey = []
timeKeyDiff = []
if (psyAvail):
    f_psy = open(itsPsyname, 'r')
    timeKey.append(datetime.timedelta()) ### append time 00:00:000.000 ?
    for line in f_psy:
        temp_psy = line.split();
        if (temp_psy[1] == "displayYUVoverlay"):
            if (temp_psy[3] == "frame"):
                frame_no = int(temp_psy[4])
                if (frame_no % 100 == 99):
                    #timestamp
                    t_s1 = temp_psy[0].split(':') # [0] = hr,  [1] = min
                    t_s2 = t_s1[2].split('.')     # [0] = sec, [1] = ms, [2] = us
                    tt = datetime.timedelta(hours=int(t_s1[0]),
                                            minutes=int(t_s1[1]),
                                            seconds=int(t_s2[0]),
                                            milliseconds=int(t_s2[1]),
                                            microseconds=int(t_s2[2]))
                    timeKey.append(tt - t2)
                    timeKeyDiff.append(timeKey[len(timeKey)-1] - timeKey[len(timeKey)-2])
                    #print "Frame " + str(frame_no) + " " + str(tt - t2) + " " + str(timeKeyDiff[len(timeKeyDiff)-1])
    f_psy.close()

timeKeyTotal = datetime.timedelta()
numtimeKey = 0
for iii in timeKeyDiff:
    timeKeyTotal += iii
    numtimeKey += 1
timeKeyAvg = timeKeyTotal / numtimeKey
print "Average time keyframes = " + str(timeKeyAvg)

### Parsing log file
print "### Start Parsing the log file ###"

### Do checking the log file for sync stuffs
tt = t0 = tl = datetime.timedelta()
timeLogKey = []
timeLogKeyDiff = []
f = open(itsFilename, 'r')
for line in f:
    tempLine = line.strip('\n')
    parseLine = tempLine.split('\t')
    ### Parsing time
    timeLine = parseLine[0].split('-')
    tt = datetime.timedelta(hours=int(timeLine[0]),
                                  minutes=int(timeLine[1]),
                                  seconds=int(timeLine[2]),
                                  milliseconds=int(timeLine[3]))
    ### Find only sync frame
    if (parseLine[1] == "Sync"):
        if (parseLine[2].strip() == "record"):
            t0 = tt
            timeLogKey.append(datetime.timedelta())
            print "Log: start recording at " + str(t0)
        elif (parseLine[2].strip() == "done"):
            tl = tt
            if (timeLogKey[len(timeLogKey)-1] != tl):
                timeLogKey.append(tl)
            print "Log: stop recording at " + str(tl)
        else:
            tn = tt - t0
            timeLogKey.append(tn)
            timeLogKeyDiff.append(timeLogKey[len(timeLogKey)-1] - \
                                  timeLogKey[len(timeLogKey)-2])
            #print "Log: event at " + str(tn) + " : " + parseLine[2].strip() + \
            #    " diff: " + str(timeLogKeyDiff[len(timeLogKeyDiff)-1])
    elif (parseLine[1] == "Joystick"): #Joystick detected
        joyEn = True
        ## Check number of button
        butL = parseLine[4].split()
        joyButNo = len(butL)
    elif (parseLine[1] == "Keyboard"): #Keyboard detected
        keyEn = True
    elif (parseLine[1] == "Mouse   "): #Mouse detected
        mouEn = True
        
f.close()

timeLogKeyTotal = datetime.timedelta()
numtimeLogKey = 0            
for jjj in timeLogKeyDiff:
    timeLogKeyTotal += jjj
    numtimeLogKey += 1
timeLogKeyAvg = timeLogKeyTotal / numtimeLogKey
print "Average time keyframes = " + str(timeLogKeyAvg)

timeTotalDiff = datetime.timedelta()
stretchFactorArr = []
for kkk in range(len(timeKeyDiff)):
    
    timeTotalDiff += (timeKeyDiff[kkk] - timeLogKeyDiff[kkk])
    psyTime = timeKeyDiff[kkk].microseconds + (timeKeyDiff[kkk].seconds*1000000)
    logTime = timeLogKeyDiff[kkk].microseconds + (timeLogKeyDiff[kkk].seconds*1000000)
    stretchFactor = float(psyTime)/float(logTime)
    stretchFactorArr.append(stretchFactor)
    #print "Difference key "+str(kkk)+" is "+str(abs(timeKeyDiff[kkk] - timeLogKeyDiff[kkk])) + " Stretch Factor = " + str(stretchFactor)
#print "Avg Difference Key = " + str(abs(timeTotalDiff) / len(timeKeyDiff))
#print stretchFactorArr

### Compute the log's running frequency
#itsPeriodic = 4.0 / 1000.0
print str(timeKeyAvg) + "_____" + str(timeLogKeyAvg)
itsPeriodicTime = timeKeyAvg.microseconds + (timeKeyAvg.seconds * 1000000)
itsPeriodic = (itsPeriodicTime * 1.0 / 1000000.0) / 50.0
#if (1/itsPeriodic) > 29.97 : ## Film max have 29.97Hz, above is abnormal
#    itsFreq = 250.25025
#else:
itsFreq = (30.0/(1.0 / itsPeriodic)) * 200.0
### FIXXXXXX:
#itsFreq = 250.0
itsTrash = 0
print "Frequency of film = " + str(1/itsPeriodic)
print "Frequency of logger = " + str(itsFreq)

### Saving extra data
if (enExtraFiles):
    cleanFile(itsRatename)
    cleanFile(itsTrashname)
    cleanFile(itsOffsetname)
    writeFile(itsRatename, str(itsFreq)+"Hz")
    #writeFile(itsRatename, "250.25025Hz") ### 29.97fps film
    #writeFile(itsRatename, "250.333778371Hz") ### 29.96fps film
    writeFile(itsTrashname, str(itsTrash))
else:
    writeFile(itsTargetname, "period="+str(itsFreq)+"Hz")
    #writeFile(itsTargetname, "period=250.25025Hz") ### due film at 29.97fps
    #writeFile(itsTargetname, "period=250.333778371Hz") ### due film at 29.96fps
    writeFile(itsTargetname, "trash="+str(itsTrash))
    writeFile(itsTargetname, "res="+itsScreenSize)
# Check column list
colLine = ""
defBlankLine = ""
if mouEn :
    colLine += "mx my ml mm mr "
    defBlankLine += "0 0 0 0 0"
if joyEn :
    colLine += "x y b" + str(joyButNo) + " "
    defBlankLine += "0 0 "
    for i in range(joyButNo):
        defBlankLine += "0 "
if keyEn :
    colLine += "k"
    defBlankLine += ""
if (colLine == "") :
    print "Nothing detected?"
    exit
writeFile(itsTargetname, "cols="+colLine)

### pre alloc variables
nowTime = datetime.timedelta()
baseTime = datetime.timedelta()
baseLogTime = datetime.timedelta()
keyTime = datetime.timedelta()
keyLogTime = datetime.timedelta()
currTime = datetime.timedelta()
prevTime = datetime.timedelta()
totalTime = datetime.timedelta()
totalGameTime = datetime.timedelta()
maxTimeLag = datetime.timedelta()
tmptime = datetime.timedelta()
totalAct = 0
totalGameAct = 0
### sync stuffs
totalFrameSync = 0
syncLineList = []
syncKeyFrame = []
syncNo = 0

joyX="0"
joyY="0"
joyBut=""
for i in range(joyButNo):
    joyBut+="0 "

mouXY="0 0"
mouBL=False
mouBM=False
mouBR=False

keyPress=[]
keyPress.append("")

tmpline = ""
#syncLineList.append(tmpline.strip())
syncLineList.append(defBlankLine)
syncKeyFrame.append(keyTime)

itsLineCompensation = round(itsCompensation * itsFreq / 30.0)
if itsLineCompensation < 1 :
    itsLineCompensation = 1

### Open the filename in read mode, and the targetfile in write mode for overwriting
f = open(itsFilename, 'r')
for line in f:
    tempLine = line.strip('\n')
    parseLine = tempLine.split('\t')
    
    
    ### Parsing time
    timeLine = parseLine[0].split('-')
    nowTime = datetime.timedelta(hours=int(timeLine[0]),
                                 minutes=int(timeLine[1]),
                                 seconds=int(timeLine[2]),
                                 milliseconds=int(timeLine[3]))
    

    ### if its a sync frame
    if (parseLine[1] == "Sync"):
        #print str(nowTime - keyTime)
        keyLogTime = nowTime
        if len(timeKey) > syncNo+1:
            keyTime = baseTime + timeKey[syncNo+1]
        #print keyTime
        #print timeKey[syncNo]
        #syncLineList.append(tmpline.strip())
        #syncKeyFrame.append(keyTime)
        #print "x="+str(len(syncLineList))+" x="+str(len(syncKeyFrame))
        #print str(nowTime) + " versus " + str(tmptime)
        if (parseLine[2].strip() == "record"):
            # start recording, all previous data are trash
            # itsTrash = totalFrameSync
            if len(syncLineList) > 0 :
                oldLineTmp = syncLineList.pop()
                for kkk in range(itsLineCompensation) :
                    writeFile(itsTargetname, oldLineTmp)
            else :
                oldLineTmp = defBlankLine
            syncLineList = []
            syncKeyFrame = []
            syncNo = -1
            #oldTime = keyTime
            #prevTime = keyTime
            baseTime = nowTime
            keyTime = baseTime
            prevTime = baseTime
            print "start"
            #print baseTime
        elif (parseLine[2].strip() == "done"):
            # use previous time diffs
            print "done"
            #for itsLine in syncLineList:
            #currTime += timeDiffFrame
            #    while (prevTime < currTime):
            #        writeFile(itsTargetname,itsLine)
            #        prevTime += datetime.timedelta(milliseconds=4)
            for i in range(1,len(syncLineList)):
                currTime = syncKeyFrame[i]
                oldLineTmp = syncLineList[i-1]
                while (prevTime < currTime):
                    writeFile(itsTargetname, syncLineList[i-1])
                    #writeFile(itsTargetname, syncLineList[i])
                    prevTime += datetime.timedelta(milliseconds=5)
        else:
            # frame syncs
            print parseLine[2].strip()
            #timeDiffFrame = timeKeyDiff[syncNo] / totalFrameSync
            #for itsLine in syncLineList:
            #    currTime += timeDiffFrame
            #    while (prevTime < currTime):
            #        writeFile(itsTargetname,itsLine)
            #        prevTime += datetime.timedelta(milliseconds=4)
            for i in range(1,len(syncLineList)):
                currTime = syncKeyFrame[i]
                #print str(prevTime)+"|"+str(currTime)
                iteration = 0
                oldLineTmp = syncLineList[i-1]
                while (prevTime < currTime):
                    writeFile(itsTargetname, syncLineList[i-1])
                    #writeFile(itsTargetname, syncLineList[i])
                    prevTime += datetime.timedelta(milliseconds=5)
                    iteration += 1
                #print str(iteration) +" : "+ syncLineList[i-1]
                
        syncLineList = []
        syncKeyFrame = []
        totalFrameSync = 0
        syncNo += 1
        syncLineList.append(oldLineTmp)
        syncKeyFrame.append(prevTime)
        #continue

    elif (parseLine[1] == "Joystick"):
        #syncLineList.append(parseLine[2]+" "+parseLine[3]+" "+parseLine[4].strip())
        joyX=parseLine[2]
        joyY=parseLine[3]
        joyBut=parseLine[4].strip()
        #totalFrameSync += 1
    elif (parseLine[1] == "Mouse   "):
        if parseLine[2] == "<MOVE>":
            mouXY = parseLine[3].strip()
        elif parseLine[2] == "<MOUSELD>":
            mouBL = True
            mouXY = parseLine[3].strip()
        elif parseLine[2] == "<MOUSELU>":
            mouBL = False
            mouXY = parseLine[3].strip()
        elif parseLine[2] == "<MOUSEMD>":
            mouBM = True
            mouXY = parseLine[3].strip()
        elif parseLine[2] == "<MOUSEMU>":
            mouBM = False
            mouXY = parseLine[3].strip()
        elif parseLine[2] == "<MOUSERD>":
            mouBR = True
            mouXY = parseLine[3].strip()
        elif parseLine[2] == "<MOUSERU>":
            mouBR = False
            mouXY = parseLine[3].strip()
        #totalFrameSync += 1
    elif (parseLine[1] == "Keyboard"):
        tmpline = parseLine[2].split()
        if tmpline[1].strip() == "down" :
            keyButPressed = False
            keytmp = tmpline[0].strip()
            keytmp = keytmp.strip("\<\>")
            for keyline in keyPress:
                if keyline == keytmp:
                    keyButPressed = True
            if keyButPressed == False:
                keyPress.append(keytmp)
        elif tmpline[1].strip() == "up" :
            keyButPressed = False
            keytmp = tmpline[0].strip()
            keytmp = keytmp.strip("\<\>")
            for keyline in keyPress:
                if keyline == keytmp:
                    keyButPressed = True
            if keyButPressed == True:
                keyPress.remove(keytmp)
        #totalFrameSync += 1
        #print keyPress

    ### Now we combine above data into a single time event
    tmpline=""
    if mouEn :
        tmpline += mouXY
        if mouBL :
            tmpline += " 1 "
        else:
            tmpline += " 0 "
        if mouBM :
            tmpline += "1 "
        else:
            tmpline += "0 "
        if mouBR :
            tmpline += "1 "
        else:
            tmpline += "0 "
        # Hey, its empty?
        if (tmpline == ""):
            tmpline = "0 0 0 0 0 "
    if joyEn :
        tmpline += joyX + " " + joyY + " " + joyBut + " "
    if keyEn :
        for keyline in keyPress:
            tmpline += keyline + " "
    syncLineList.append(tmpline.strip())
    
    ### now counting the event time
    if syncNo >= len(stretchFactorArr):
        tmpstretch = 1.0
    else:
        tmpstretch = stretchFactorArr[syncNo]
    tmptime = rescaleTime(keyTime,keyLogTime, nowTime, tmpstretch)
    #print str(tmptime)+"|"+str(keyTime)+"|"+str(keyLogTime)+"|"+str(nowTime)
    syncKeyFrame.append(tmptime)

    #print str(tmptime) + " - " + tmpline.strip()

### We are done reading the file. close it
f.close()

