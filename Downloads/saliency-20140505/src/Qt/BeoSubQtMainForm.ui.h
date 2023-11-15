/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you want to add, delete, or rename functions or slots, use
** Qt Designer to update this file, preserving your code.
**
** You should not define a constructor or destructor in this file.
** Instead, write your code in functions called init() and destroy().
** These will automatically be called by the form's constructor and
** destructor.
*****************************************************************************/

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

void BeoSubQtMainForm::init(const nub::soft_ref<BeoSubOneBal>& sub,
                            const nub::soft_ref<Beowulf>& beo,
                            const nub::soft_ref<BeoSubTaskDecoder>& decoder,
                            const nub::soft_ref<ColorTracker>& tracker1,
                            const nub::soft_ref<ColorTracker>& tracker2,
                            const nub::soft_ref<BeoSubCanny>& detector)
{
    DIR *dir;
    struct dirent *entry;
    int size = 50;
    float hp, hi, hd;
    float pp, pi, pd;
    float dp, di, dd;
    char buffer[size], fileName[20];

    counterUp = counterFront = counterDown = 0;
    if (getcwd(buffer, size) == NULL) PLFATAL("error in getcwd");
    if((dir = opendir(buffer)) == NULL)
        printf("open dir error");
    else
    {
        while((entry = readdir(dir)) !=NULL)
        {
            if(entry->d_name[0] != '.')
            {
                strcpy(fileName, entry->d_name);
                QString qs(fileName);
                if(qs.contains("top") > 0)
                {
                    topFiles.push_back(fileName);
                    qs.replace("top", "");
                    qs.replace(".txt", "");
                    if(qs.toInt() > counterUp)
                        counterUp = qs.toInt();
                }
                if(qs.contains("front") > 0)
                {
                    frontFiles.push_back(fileName);
                    qs.replace("front", "");
                    qs.replace(".txt", "");
                    if(qs.toInt() > counterFront)
                        counterFront = qs.toInt();
                }
                if(qs.contains("bottom") > 0)
                {
                    bottomFiles.push_back(fileName);
                    qs.replace("bottom", "");
                    qs.replace(".txt", "");
                    if(qs.toInt() > counterDown)
                        counterDown = qs.toInt();
                }
            }
        }
    }
    if(counterUp != 0)
        counterUp++;
    if(counterFront != 0)
        counterFront++;
    if(counterDown != 0)
        counterDown++;
    joyflag = true;
  itsSub = sub; itsBeo = beo; itsTracker1 = tracker1; itsTracker2 = tracker2; itsDecoder = decoder; itsDetector = detector;
  FrontBDisplay->setText(QString(convertToString(0)));
  RearBDisplay->setText(QString(convertToString(0)));
  BsDisplay->setText(QString(convertToString(0)));

  LTDisplay->setText(QString(convertToString(0)));
  RTDisplay->setText(QString(convertToString(0)));
  TsDisplay->setText(QString(convertToString(0)));

  TurnDisplay->setText(QString(convertToString(0)));
  TiltDisplay->setText(QString(convertToString(0)));
  savePIDgain = fopen("PIDgains.pid", "r");
  if (fscanf(savePIDgain, "%f %f %f\n", &hp, &hi, &hd) != 3) LFATAL("fscanf error");
  if (fscanf(savePIDgain, "%f %f %f\n", &pp, &pi, &pd) != 3) LFATAL("fscanf error");
  if (fscanf(savePIDgain, "%f %f %f\n", &dp, &di, &dd) != 3) LFATAL("fscanf error");
  HeadingP->setText(QString(sformat("%f",hp)));
  HeadingI->setText(QString(sformat("%f",hi)));
  HeadingD->setText(QString(sformat("%f",hd)));
  PitchP->setText(QString(sformat("%f",pp)));
  PitchI->setText(QString(sformat("%f",pi)));
  PitchD->setText(QString(sformat("%f",pd)));
  DepthP->setText(QString(sformat("%f",dp)));
  DepthI->setText(QString(sformat("%f",di)));
  DepthD->setText(QString(sformat("%f",dd)));
  fclose(savePIDgain);
  startTimer(1000);  // start a timer for our displaay refreshing

}

void BeoSubQtMainForm::togglePic()
{
    takePic = !takePic;
}
void BeoSubQtMainForm::savePIDvalue()
{
  savePIDgain = fopen("PIDgains.pid", "w");
  fprintf(savePIDgain, "%f %f %f\n", HeadingP->text().toFloat(), HeadingI->text().toFloat(), HeadingD->text().toFloat());
  fprintf(savePIDgain, "%f %f %f\n", PitchP->text().toFloat(), PitchI->text().toFloat(), PitchD->text().toFloat());
  fprintf(savePIDgain, "%f %f %f", DepthP->text().toFloat(), DepthI->text().toFloat(), DepthD->text().toFloat());
  fclose(savePIDgain);
}
void BeoSubQtMainForm::HeadingPID_toggled()
{
    if(HeadingPID->isOn())
    {
        itsSub->setHeadingPgain(HeadingP->text().toFloat());
        itsSub->setHeadingIgain(HeadingI->text().toFloat());
        itsSub->setHeadingDgain(HeadingD->text().toFloat());
        itsSub->useHeadingPID(true);
        savePIDvalue();
    }
    else
        itsSub->useHeadingPID(false);
}

void BeoSubQtMainForm::PitchPID_toggled()
{
    if(PitchPID->isOn())
    {
        itsSub->setPitchPgain(PitchP->text().toFloat());
        itsSub->setPitchIgain(PitchI->text().toFloat());
        itsSub->setPitchDgain(PitchD->text().toFloat());
        itsSub->usePitchPID(true);
        savePIDvalue();
    }
    else
        itsSub->usePitchPID(false);
}

void BeoSubQtMainForm::DepthPID_toggled()
{
    if(DepthPID->isOn())
    {
        itsSub->setDepthPgain(DepthP->text().toFloat());
        itsSub->setDepthIgain(DepthI->text().toFloat());
        itsSub->setDepthDgain(DepthD->text().toFloat());
        itsSub->useDepthPID(true);
        savePIDvalue();
    }
    else
        itsSub->useDepthPID(false);
}

void BeoSubQtMainForm::KILL_toggled()
{
    if(KILL->isOn())
        itsSub->useKillSwitch(true);
    else
        itsSub->useKillSwitch(false);
}

void BeoSubQtMainForm::turnopen_return()
{
    Angle a = double(to->text().toDouble());
    itsSub->turnOpen(a, true);
}

void BeoSubQtMainForm::advance_return()
{
    itsSub->advanceRel(lineEditAdvance->text().toFloat());
}

void BeoSubQtMainForm::strafe_return()
{
    itsSub->strafeRel(lineEditStrafe->text().toFloat());
}

void BeoSubQtMainForm::resetAtt_return()
{
  LERROR("unimplemented");
}

void BeoSubQtMainForm::FrontBallast_valueChanged(int val)
{
  // parse the val between 0 and 1
  itsSub->setFrontBallast(float(val)/100.0F);
  FrontBDisplay->setText(QString(convertToString(val)));
}

void BeoSubQtMainForm::RearBallast_valueChanged(int val)
{
  // parse the val between 0 and 1
  itsSub->setRearBallast(float(val)/100.0F);
  RearBDisplay->setText(QString(convertToString(val)));
}

void BeoSubQtMainForm::LThruster_valueChanged(int val)
{
  float rt, lt;
  itsSub->getThrusters(lt, rt);
  // parse the val between  -1 and 1
  itsSub->thrust(float(val)/100.0F, rt);
  LTDisplay->setText(QString(convertToString(val)));
}

void BeoSubQtMainForm::RThuster_valueChanged(int val)
{
  float rt, lt;
  itsSub->getThrusters(lt, rt);
  // parse the val between  -1 and 1
  itsSub->thrust(lt, float(val)/100.0F);
  RTDisplay->setText(QString(convertToString(val)));
}

void BeoSubQtMainForm::Thrusters_valueChanged(int val)
{
  // parse the val between  -1 and 1
  itsSub->thrust(float(val)/100.0F, float(val)/100.0F);
  LTScroll->setValue(val);
  RTScroll->setValue(val);
  TsDisplay->setText(QString(convertToString(val)));
}

void BeoSubQtMainForm::dive_valueChanged(int val)
{
  itsSub->diveAbs(float(val)/10.0F, false);
  BsDisplay->setText(QString(convertToString(val)));
}

void BeoSubQtMainForm::Orient_valueChanged(int val)
{
  Angle a = double(val);
  itsSub->turnAbs(a, false);
  TurnDisplay->setText(QString(convertToString(val)));
}

void BeoSubQtMainForm::Pitch_valueChanged(int val)
{
  Angle a = double(val);
  itsSub->pitchAbs(a, false);
  TiltDisplay->setText(QString(convertToString(val)));
}

void BeoSubQtMainForm::timerEvent( QTimerEvent *e )
{
  displayFunc();
}

void BeoSubQtMainForm::saveImageLog(const char* name, int count)
{
    FILE *f;
    float lt,rt;
    std::string s;
    itsSub->getThrusters(lt, rt);
    s = sformat("%s%06d.txt", name, count);
    //NOTE: there is more info than this available in Attitude.
    f = fopen(s.c_str(), "w");
    fprintf(f, "%s%06d.png\t", name, count);
    fprintf(f, "%f\t",itsSub->getHeading().getVal());
    fprintf(f, "%f\t",itsSub->getPitch().getVal());
    fprintf(f, "%f\t",itsSub->getRoll().getVal());
    fprintf(f, "%f\t",itsSub->getDepth());
    fprintf(f, "%f\t",itsSub->getFrontBallast());
    fprintf(f, "%f\t",itsSub->getRearBallast());
    fprintf(f, "%f\t",lt);
    fprintf(f, "%f\t",rt);
    fprintf(f, "-1000 -1000\t");
    fprintf(f, "%f", itsSub->getTime());
    // fprintf(f, "XACCel:%f\n", itsSub->itsIMU->getXaccel());
    //fprintf(f, "\nxvel:%f\n", itsSub->itsIMU->getXvel().getVal);
    //fprintf(f, "yvel:%f\n", itsSub->itsIMU->getYvel().getVal);
    //fprintf(f, "Zvel:%f\n", itsSub->itsIMU->getZvel().getVal);
    //fprintf(f, "XACCel:%f\n", itsSub->itsIMU->getXaccel());
    //fprintf(f, "YACCel:%f\n", itsSub->itsIMU->getYaccel());
    //fprintf(f, "ZACCel:%f\n", itsSub->itsIMU->getZaccel());
    fprintf(f, "\n");
    fclose(f);

}

void BeoSubQtMainForm::saveImageUp()
{
  // save image
  Raster::WriteRGB( imup, sformat("top%06d.png", counterUp) );
  topFiles.push_back(sformat("top%06d", counterUp));

  // the log file for the image
  saveImageLog("top", counterUp);
  counterUp++;
}

void BeoSubQtMainForm::saveImageFront()
{
  //save image
  Raster::WriteRGB( imfront, sformat("front%06d.png", counterFront) );
  frontFiles.push_back(sformat("front%06d", counterFront));

  // the log file for the image
  saveImageLog("front", counterFront);
  counterFront++;
}

void BeoSubQtMainForm::saveImageDown()
{
  // save image
  Raster::WriteRGB( imdown, sformat("bottom%06d.png", counterDown) );
  bottomFiles.push_back(sformat("bottom%06d", counterDown));

  // the log file for the image
  saveImageLog("bottom", counterDown);
  counterDown++;
}

void BeoSubQtMainForm::taskGate(){
  printf("You wish this were implemented\n");
}

void BeoSubQtMainForm::taskDecode()
{
    ImageSet< PixRGB<byte> > inStream;
    Image< PixRGB<byte> > img;
    int NAVG = 20; Timer tim; uint64 t[NAVG]; int frame = 0;
    float avg2 = 0.0;
    //Grab a series of images
    for(int i = 0; i < 100; i++){
        tim.reset();

        img = itsSub->grabImage(BEOSUBCAMFRONT);
        inStream.push_back(img);

        uint64 t0 = tim.get();  // to measure display time
        t[frame % NAVG] = tim.get();
        t0 = t[frame % NAVG] - t0;
        // compute framerate over the last NAVG frames:
        if (frame % NAVG == 0 && frame > 0)
        {
            uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
            avg2 = 1000.0F / float(avg) * float(NAVG);
        }
        frame ++;
    }

    itsDecoder->setupDecoder("Red", false);
    itsDecoder->runDecoder(inStream, avg2);
    float hertz = itsDecoder->calculateHz();
    printf("\n\nFound Red with %f Hz\n\n", hertz);
}

void BeoSubQtMainForm::taskA()
{
  int counter = 0;
  while(counter > 8 && itsSub->getDepth() < 3.0){
    if(itsSub->affineSIFT(BEOSUBCAMDOWN, itsSub->itsVOtaskAdown)){//approach
      itsSub->diveAbs(3.0);
    }
  }
  /*
  float tol = 20.0;
  bool notDone = true;
  bool findGot = false;
  while(notDone){
    if(itsSub->centerColor("Orange", BEOSUBCAMFRONT, tol)){
      if(itsSub->centerColor("Orange", BEOSUBCAMDOWN, tol)){
        printf("\n\nDIVINGn\n");
        itsSub->diveAbs(2.0);//uberdive FIX?
        findGot = true;
      }
    }
    else{
      if(!itsSub->centerColor("Orange", BEOSUBCAMDOWN, tol)){
        itsSub->thrust(-0.6, 0.6);//spin to hit it
        printf("spinn\n");
      }
      else{
        findGot = true;
        itsSub->diveAbs(2.0);

      }
    }
    if(findGot && (itsSub->getDepth() >= 0.8)){// CHANGE VALUE FOR FINAL! FIX!
      printf("DONE!\n");
      notDone = false;
      //diveAbs(3.0);
    }
  }
  */
}

void BeoSubQtMainForm::taskB(){
  if(itsSub->TaskB())
    printf("\n\nFINISHED TASK B!\n\n");//wishful thinking
  else
          printf("faild TASK B\n");
 }

void BeoSubQtMainForm::taskC(){

}

void BeoSubQtMainForm::stopAll()
{
    //NOTE: Some of this may be bunk now. FIX!
  itsSub->thrust(0, 0);
  itsSub->setFrontBallast(0.0);
  itsSub->setRearBallast(0.0);
  FrontBScroll->setValue(0);
  RearBScroll->setValue(0);
  DiveScroll->setValue(0);
  LTScroll->setValue(0);
  RTScroll->setValue(0);
  TsScroll->setValue(0);
}

void BeoSubQtMainForm::matchAndDirect()
{
    Image< PixRGB<byte> > img = imdown;
    BeoSubDB* db = new BeoSubDB;
    db->initializeDatabase(bottomFiles, "bottomDB.txt");
    VisualObjectDB vdb;
    if(!vdb.loadFrom("bottomVDB.txt")){
        printf("Must create Visual Object Database befirehand\n");
        return;
    }
    rutz::shared_ptr<VisualObject> vo(new VisualObject("matching.png", "matching.png", imdown));
    std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
    const uint nmatches = vdb.getObjectMatches(vo, matches, VOMA_KDTREEBBF);

    if(nmatches == 0U){
        printf("No match found\n");
        return;
    }
    rutz::shared_ptr<VisualObject> obj = matches[0]->getVoTest();
    std::string foundName(obj->getName().c_str());

    uint idx = foundName.rfind('.'); foundName = foundName.substr(0, idx);
    foundName = foundName + ".txt";

    printf("Found match %s with score %f\n", foundName.c_str(), matches[0]->getScore());
    MappingData foundMD = db->getMappingData(foundName);
    MappingData goalMD = db->getMappingData("TaskA.txt"); //Assumes file exists for TaskA!

    Attitude directions;
    float distance = 0.0;
    db->getDirections(foundMD, goalMD, directions, distance);

    printf("\n\nThe needed depth is %f, heading is %f, and distance is %f\n\n", directions.depth, directions.heading.getVal(), distance);
    return;
}

void BeoSubQtMainForm::parseMessage(TCPmessage& rmsg)
{
  float yAxis = 0.0;
  float xAxis = 0.0;
  float xL=0.0F, xR=0.0F;
  float yL=0.0F, yR=0.0F;
  int button1 = 0;
  int button2 = 0;
  int button3 = 0;

  //Parse message
  //order of dimensions: yAxis, zAxis, slider, button1, button2
  Image<float> in = rmsg.getElementFloatIma();

  //Y-AXIS = Advance
  yAxis = in.getVal(0);
  //X-AXIS = Turn
  xAxis = in.getVal(1);
  //SLIDER
  //BUTTON 1
  button1 = (int)in.getVal(3);
  //BUTTON 2
  button2 = (int)in.getVal(4);
  //BUTTON 3
  button3 = (int)in.getVal(5);

  if( yAxis > 0 )
  {
      // forward
      yL = 1.0F - 1.0F/32767 * yAxis; yR = 1.0F - 1.0F/32767 * yAxis;
      joyflag =true;
  }
  else if(yAxis < 0)
  {
      //backward
      yL = -1.0F - 1.0F/32767*yAxis; yR = -1.0F - 1.0F/32767*yAxis;
      joyflag = false;
  }
  else if(yAxis == 0)
  {
      if(joyflag)
      {
          yL = 1.0F; yR = 1.0F;
      }
      else
      {
          yL = -1.0F; yR = -1.0F;
      }
  }
  if( xAxis >0 )
    {
      // turn right
      xL = -1.0F + 1.0F/32767*xAxis; xR = 1.0F - 1.0F/32767*xAxis;
    }
  else if(xAxis < 0)
    {
      // turn left
      xL = 1.0 + 1.0/32767*xAxis; xR =  -1.0 - 1.0/32767*xAxis;
    }

  if(xR == 0.0F && xL == 0.0F)
  {
      itsSub->thrust(yL, yR);
      LTDisplay->setText(QString(sformat("%.3f", yL)));
      RTDisplay->setText(QString(sformat("%.3f", yR)));
  }
  else if(yR == 0.0F && yL == 0.0F)
  {
      itsSub->thrust(xL, xR);
      LTDisplay->setText(QString(sformat("%.3f", xL)));
      RTDisplay->setText(QString(sformat("%.3f", xR)));
  }
  else
  {
      itsSub->thrust( xL+yL / 2.0F, xR+yR / 2.0F);
      LTDisplay->setText(QString(sformat("%.3f", xL+yL / 2.0F)));
      RTDisplay->setText(QString(sformat("%.3f", xR+yR / 2.0F)));
  }
  if(button1 == 1.0F)
          saveImageUp();
  else if(button2 == 1.0F)
          saveImageFront();
  else if(button3 == 1.0F)
          saveImageDown();

}

void BeoSubQtMainForm::displayFunc()
{
    Image< PixRGB<byte> > img;
    QPixmap qpixm;
    float lt,rt;
    FILE *f;
    char temp[50];
    int tempNum;

    f = fopen("/proc/acpi/thermal_zone/THRM/temperature", "r");
    if (f == NULL)
      CPUtemp->setText(QString("can't find the file"));
    else
      {
        if (fscanf(f, "%s%d", temp, &tempNum) != 2) LFATAL("fscanf error");
        CPUtemp->setText(QString(convertToString(tempNum)));
        fclose(f);
      }

    if(itsSub->targetReached())
        indicator->setText(QString("On Target"));
    else
        indicator->setText(QString("Homing"));


    imup = itsSub->grabImage(BEOSUBCAMUP);
    qpixm = convertToQPixmap(imup);
    ImagePixmapLabel1->setPixmap( qpixm );

    imfront = itsSub->grabImage(BEOSUBCAMFRONT);
    qpixm = convertToQPixmap(imfront);
    ImagePixmapLabel2->setPixmap( qpixm );

    imdown = itsSub->grabImage(BEOSUBCAMDOWN);
    qpixm = convertToQPixmap(imdown);
    ImagePixmapLabel3->setPixmap( qpixm );

    itsSub->getThrusters(lt, rt);

    LineEdit1->setText(QString(sformat("%.3f", itsSub->getRoll().getVal())));
    LineEdit2->setText(QString(sformat("%.3f", itsSub->getPitch().getVal())));
    LineEdit3->setText(QString(sformat("%.3f", itsSub->getHeading().getVal())));
    LineEdit4->setText(QString(sformat("%.3f", itsSub->getDepth())));

    LineEdit1_2->setText(QString(sformat("%.3f", itsSub->getTargetAttitude().roll.getVal())));
    LineEdit2_2->setText(QString(sformat("%.3f", itsSub->getTargetAttitude().pitch.getVal())));
    LineEdit3_2->setText(QString(sformat("%.3f", itsSub->getTargetAttitude().heading.getVal())));
    LineEdit4_2->setText(QString(sformat("%.3f", itsSub->getTargetAttitude().depth)));

    LineEdit11->setText(QString(sformat("%.3f", itsSub->getFrontBallast())));
    LineEdit12->setText(QString(sformat("%.3f", itsSub->getRearBallast())));
    LineEdit15->setText(QString(sformat("%.3f", lt)));
    LineEdit16->setText(QString(sformat("%.3f", rt)));
    topfilename->setText(QString(sformat("top%06d.png",counterUp)));
    frontfilename->setText(QString(sformat("front%06d.png",counterFront)));
    bottomfilename->setText(QString(sformat("bottom%06d.png",counterDown)));
    // Networking code here
    if (itsBeo.isValid())
      {
        int32 rframe, raction, rnode = -1;  // receive from any node
        for (int ii = 0; ii < 30; ii ++)
          {
            if (itsBeo->receive(rnode, rmsg, rframe, raction))
              parseMessage(rmsg);
          }
      }
    if(takePic)
    {
      saveImageUp();
      saveImageFront();
      saveImageDown();
  }

  }

void BeoSubQtMainForm::keyPressEvent( QKeyEvent *event )
{
    switch(event->key())
    {
    // stop all
    case  Key_Space:
        //stopAll();
        itsSub->thrust(0.0, 0.0);
        break;
    // dive
    case Key_A:
        if(itsSub->getFrontBallast() + 0.05F <= 0.95F)
            itsSub->setFrontBallast(itsSub->getFrontBallast() + 0.05F);
        else
            itsSub->setFrontBallast(1.0F);

        if(itsSub->getRearBallast() + 0.05F <= 0.95F)
            itsSub->setRearBallast(itsSub->getRearBallast() + 0.05F);
        else
            itsSub->setRearBallast(1.0F);

        break;
    // surface
    case Key_Q:
        if(itsSub->getFrontBallast() - 0.05F >= 0.05F)
            itsSub->setFrontBallast(itsSub->getFrontBallast() - 0.05F);
        else
            itsSub->setFrontBallast(0.0F);

        if(itsSub->getRearBallast() - 0.05F >= 0.05F)
            itsSub->setRearBallast(itsSub->getRearBallast() - 0.05F);
        else
            itsSub->setRearBallast(0.0F);
        break;

    case Key_Z:
        saveImageUp();
        break;
    case Key_X:
        saveImageFront();
        break;
    case Key_C:
        saveImageDown();
        break;

    // advance
    case Key_I:
        itsSub->thrust (1.0, 1.0);
        break;
    // turn left
    case Key_J:
        itsSub->thrust(-1.0, 1.0);
        break;
    // backward
    case Key_K:
        itsSub->thrust(-1.0,-1.0);
        break;
    // turn right
    case Key_L:
        itsSub->thrust( 1.0,-1.0);
        break;
    }
}

void BeoSubQtMainForm::keyReleaseEvent( QKeyEvent *event )
{
    switch(event->key())
    {
    case Key_I:
    case Key_J:
    case Key_K:
    case Key_L:
        itsSub->thrust(0.0,0.0);
        break;
    }
}
