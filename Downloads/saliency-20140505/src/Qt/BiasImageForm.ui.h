/*! @file Qt/BiasImageForm.ui.h Main form for viewing smap and channels */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/BiasImageForm.ui.h $
// $Id: BiasImageForm.ui.h 14376 2011-01-11 02:44:34Z pez $

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

#include "Channels/OptimalGains.H"
#include "Channels/SubmapAlgorithmBiased.H"
#include "ObjRec/BayesianBiaser.H"
#include "Neuro/NeuroSimEvents.H"
#include "Media/MediaSimEvents.H"
#include "ObjRec/MaskBiaser.H"
#include "GUI/DebugWin.H"
#include "Image/MathOps.H"

void BiasImageForm::fileOpen()
{

  QString file = QFileDialog::getOpenFileName( "xmlFiles/all.xml",
      "ImageSet (*.xml)",
      this, "OpenImageDialog",
      "Choose Image" );
  if( !file.isEmpty() ) {
    itsTestScenes = new TestImages(file.ascii(), TestImages::XMLFILE );
    itsCurrentScene = 0;
    itsImg = itsTestScenes->getScene(itsCurrentScene); //load scene 0 by default
    itsSceneNumber->setMaxValue(itsTestScenes->getNumScenes());
    imgDisp->setImage(itsImg);

    //fill in the objects
    itsSelectObject->clear();
    for (uint obj=0; obj<itsTestScenes->getNumObj(itsCurrentScene); obj++) //look at all the objects
    {
      TestImages::ObjData objData = itsTestScenes->getObjectData(itsCurrentScene, obj, true);

      itsSelectObject->insertItem(QString(objData.description));
    }


    itsNewImage = true;
  }
}


void BiasImageForm::fileSave()
{

}


void BiasImageForm::fileExit()
{
}


void BiasImageForm::init( ModelManager & manager, nub::ref<Brain> brain,
                          nub::ref<SimEventQueue> seq)
{

  itsMgr = &manager;

  //Init the descriptor vector
  itsBrain = brain;
  itsSEQ = seq;

  //Build the descriptor vector

  //generate a complexChannel from Visual cortex to pass to DescriptorVec

  LFATAL("descriptor vecs should be handled via a SimReq to VisualCortex, which will then propagate to the channels");
  /*
  ComplexChannel *cc =
    &*dynCastWeak<ComplexChannel>(itsBrain->getVC());

  itsDescriptorVec = new DescriptorVec(*itsMgr,
      "Descriptor Vector", "DecscriptorVec", cc);

  //set up featuers and 2 classes
  itsBayesNetwork = new Bayes(itsDescriptorVec->getFVSize(), 0);
  itsDescriptorVecDialog.init(*itsDescriptorVec);

  //set the feature names
  for (uint i = 0; i < cc->numSubmaps(); i++)
  {
    const std::string name = cc->getSubmapName(i);
    itsBayesNetwork->setFeatureName(i, name.c_str());
  }
  */

  //itsTrainScene = new SceneGenerator(SceneGenerator::ALOI_OBJECTS, 500, 500);
  //itsTestScene = new SceneGenerator(SceneGenerator::ALOI_OBJECTS, 500, 500);

  clickClass = -1;

}


void BiasImageForm::showBiasSettings()
{
  itsBiasSettingsDialog.init(*itsMgr);
  itsBiasSettingsDialog.show();

}


void BiasImageForm::evolveBrain()
{
  if (itsMgr->started()){

    if (itsImg.initialized() && itsNewImage)
    {
      rutz::shared_ptr<SimEventInputFrame>
        e(new SimEventInputFrame(itsBrain.get(), GenericFrame(itsImg), 0));
      itsSEQ->post(e); //post the image to the brain
      itsDescriptorVec->setInputImg(itsImg);
      itsNewImage = false;
    }


    SimTime end_time = itsSEQ->now() + SimTime::MSECS(3.0);

    while (itsSEQ->now() < end_time)
    {
      // Any new WTA winner?
      if (SeC<SimEventWTAwinner> e = itsSEQ->check<SimEventWTAwinner>(itsBrain.get()))
      {
        const Point2D<int> winner = e->winner().p;
        itsCurrentWinner = winner;
        LINFO("winner\n");
        classifyFovea(winner.i, winner.j,-1);


        if (true) showSMap();
        if (true) showTraj();
        //if (true) showChannels();

        itsSEQ->evolve();
        return;   //return after one saccade

      }


      itsSEQ->evolve();

    }
  }

}

void BiasImageForm::showTraj(){
  static bool setupTab = true;
  static ImageCanvas* disp = NULL;
  static Point2D<int> lastWinner = itsCurrentWinner;

  if (!itsOutputImg.initialized())
    itsOutputImg = itsImg;


  drawCircle(itsOutputImg, itsCurrentWinner, 50, PixRGB<byte>(255,0,0));
  drawCircle(itsOutputImg, itsCurrentWinner, 3, PixRGB<byte>(255,0,0),3);
  drawLine(itsOutputImg, lastWinner, itsCurrentWinner, PixRGB<byte>(255,0,0),1);
  lastWinner = itsCurrentWinner;

  if (setupTab){

    QWidget* tab = new QWidget(tabDisp);
    QHBoxLayout* tabLayout = new QHBoxLayout(tab, 11, 6);
    disp = new ImageCanvas(tab);
    tabLayout->addWidget(disp);
    tabDisp->insertTab(tab, "Trajectory" );
    setupTab = false;

    //set bias connection
    connect(disp, SIGNAL(mousePressed(int,int) ),
        this, SLOT( getDescriptor(int,int) ) );

  }

  if (disp)
    disp->setImage(itsOutputImg);

}


void BiasImageForm::showSMap()
{
  static bool setupTab = true;
  static ImageCanvas* disp = NULL;

  if (setupTab){
    QWidget* tab = new QWidget(tabDisp);
    QHBoxLayout* tabLayout = new QHBoxLayout(tab, 11, 6);
    disp = new ImageCanvas(tab);
    tabLayout->addWidget(disp);
    tabDisp->insertTab(tab, "SMap" );
    setupTab = false;
  }

  if (disp) {
    Image<PixRGB<byte> > img;

    LINFO("Show SMap\n");
    if (SeC<SimEventSaliencyMapOutput> smo =
        itsSEQ->check<SimEventSaliencyMapOutput>(itsBrain.get(), SEQ_ANY))
    {
      Image<float> tmp = smo->sm();
      inplaceNormalize(tmp, 0.0F, 255.0F);
      img = toRGB(rescale(tmp, tmp.getWidth()*16, tmp.getHeight()*16));
    }

    disp->setImage(img);
  }

}


void BiasImageForm::showChannels()
{
  LFATAL("update to use a SimReq for ChannelMaps -- see SimulationViewerStd");
  /*
  static bool setupTab = true;
  static ImageCanvas* disp[10];
  int numChans = itsBrain->getVC()->numChans();

  if (setupTab){
    for (int i=0; i<numChans; i++)
    {
      QWidget* tab = new QWidget(tabDisp);
      QHBoxLayout* tabLayout = new QHBoxLayout(tab, 11, 6);
      disp[i] = new ImageCanvas(tab);
      tabLayout->addWidget(disp[i]);
      tabDisp->insertTab(tab,
          itsBrain->getVC()->subChan(i)->descriptiveName());
      setupTab = false;
    }
  }

  for( int i=0; i<numChans; i++){
    Image<PixRGB<byte> > img = toRGB(itsBrain->getVC()->subChan(i)->getOutput());
    disp[i]->setImage(img);
  }
  */
}


void BiasImageForm::configureView( QAction * action )
{
  LINFO("Configure view %i %i %i", itsViewTraj, itsViewSMap, itsViewChannels);
  if (viewTrajAction->isOn()) itsViewTraj = true; else itsViewTraj = false;
  if (viewSMapAction->isOn()) itsViewSMap = true; else itsViewSMap = false;
  if (viewChannelsAction->isOn()) itsViewChannels = true; else itsViewChannels = false;
}


void BiasImageForm::getDescriptor( int x, int y, int button )
{
  LFATAL("fixme");
  /*
  ComplexChannel* cc = &*dynCastWeak<ComplexChannel>(itsBrain->getVC());
  //check if the itsBrain has evolved
  if (!cc->hasInput())
    evolveBrain();

  //Classify the object under the fovea
  if (cc->hasInput()) //check if we got an input (could be a blank image)
  {
    clickClass = 0; //let the system knwo we want to learn the featuers as target
    classifyFovea(x, y, button);
  }
  */
}

void BiasImageForm::showDescriptorVec()
{
  itsDescriptorVecDialog.show();
}


void BiasImageForm::classifyFovea(int x, int y, int button)
{
  //build descriptive vector
  itsDescriptorVec->setFovea(Point2D<int>(x,y));
  itsDescriptorVec->buildRawDV();
  LINFO("Update Descriptior %i\n", button);
  itsDescriptorVecDialog.update();

  //get the resulting feature vector
  std::vector<double> FV = itsDescriptorVec->getFV();

 // for(uint i=0; i<FV.size(); i++)
 //  LINFO("FV: %f", FV[i]);

  int cls = itsBayesNetwork->classify(FV);

  if (cls != -1)
  {
    std::string clsName(itsBayesNetwork->getClassName(cls));
    msgLabel->setText(QString("Object is %L1").arg(clsName));
    LINFO("Object belong to class %i %s", cls,clsName.c_str());
    if (1)
      logFixation(clsName.c_str(), x, y, FV);
  } else {
    LINFO("Unknown Object");
  }


  if (!editConfigureTestAction->isOn()) { //Train Mode

    if (button == 4)
    {
      QString objName = itsObjectName->text().lower();
      LINFO("Learning %s\n", objName.ascii());
      msgLabel->setText(QString("Learning %L1").arg(objName));
      itsBayesNetwork->learn(FV, objName.ascii());

      if (1)
        logFixation(objName.ascii(), x, y, FV);
    }

  } else { //Test Mode
  }



  //write stats on window
 // msgLabel->setText(
 //     QString("correct Matches(T:%L1/D:%L2) %L3/%L4 %L5% class= %L5")
 //     .arg(correctTargetMatches)
 //     .arg(correctDistracterMatches)
 //     .arg(correctTargetMatches + correctDistracterMatches)
 //     .arg(totalObjClassified)
 //     .arg(int((float(correctTargetMatches+correctDistracterMatches)/float(totalObjClassified))*100))
 //     .arg(cls));

  }




void BiasImageForm::run()
{
  for (int i=0; i<timesSpinBox->value(); i++){
  }

}


void BiasImageForm::loadBayesNetwork()
{
  QString file = QFileDialog::getOpenFileName( QString::null,
      "Bayes network (*.net)",
      this, "OpenImageDialog",
      "Choose Bayes Network" );
  if( !file.isEmpty() )
    itsBayesNetwork->load(file.ascii());

}


void BiasImageForm::saveBayesNetwork()
{
  QString file = QFileDialog::getSaveFileName( QString::null,
      "Bayes network (*.net)",
      this, "SaveImageDialog",
      "Choose Bayes Network" );
  if( !file.isEmpty() )
    itsBayesNetwork->save(file.ascii());


}


void BiasImageForm::viewBayesNetwork()
{
  itsBayesNetworkDialog.show();
  itsBayesNetworkDialog.init(*itsBayesNetwork);

}


void BiasImageForm::setBiasImage( bool biasVal )
{
  LFATAL("fixme");
  /*
  LINFO("Bias Image %i\n", biasVal);
  ComplexChannel* cc = &*dynCastWeak<ComplexChannel>(itsBrain->getVC());


  if (biasVal)
  {
    //Get the object we are biasing for
    QString objName = itsObjectName->text().lower();
    int cls = itsBayesNetwork->getClassId(objName.ascii());

    if (cls != -1)
    {
      msgLabel->setText(QString("Biasing for %L1").arg(objName));
      //Set mean and sigma to bias submap
      BayesianBiaser bb(*itsBayesNetwork, cls, -1, biasVal);
      // with distractor: BayesianBiaser bb(*itsBayesNetwork, 0, 1, biasVal);
      cc->accept(bb);

      //set the bias
      setSubmapAlgorithmBiased(*cc);
    } else {
      msgLabel->setText(QString("%L1 is not known").arg(objName));
    }
  } else {
    msgLabel->setText(QString("Unbiasing"));
    //Set mean and sigma to bias submap
    BayesianBiaser bb(*itsBayesNetwork, 0, -1, biasVal);
    // with distractor: BayesianBiaser bb(*itsBayesNetwork, 0, 1, biasVal);
    cc->accept(bb);

    //set the bias
    setSubmapAlgorithmBiased(*cc);
  }
  */
}


void BiasImageForm::showSceneSettings()
{
  //itsSceneSettingsDialog.init(itsTrainScene, itsTestScene);
  //itsSceneSettingsDialog.show();

}


void BiasImageForm::getScene( int scene )
{

  //save the image of the old scene
  if (itsOutputImg.initialized())
  {
    char filename[255];
    TestImages::SceneData sceneData = itsTestScenes->getSceneData(itsCurrentScene);
    sprintf(filename, "%s.fix.ppm", sceneData.filename.c_str());
    Raster::WriteRGB(itsOutputImg, filename);
  }


    LINFO("get Scene\n");
    itsCurrentScene = scene;
    itsImg = itsTestScenes->getScene(itsCurrentScene); //load scene 0 by default
    itsSceneNumber->setMaxValue(itsTestScenes->getNumScenes());
    Image<PixRGB<byte> > tmp = itsImg;
    if (viewShow_LabelsAction->isOn())
      itsTestScenes->labelScene(scene, tmp);
    imgDisp->setImage(tmp);
    itsNewImage = true;

    //fill in the objects
    itsSelectObject->clear();
    for (uint obj=0; obj<itsTestScenes->getNumObj(itsCurrentScene); obj++) //look at all the objects
    {
      TestImages::ObjData objData = itsTestScenes->getObjectData(itsCurrentScene, obj, true);

      itsSelectObject->insertItem(QString(objData.description));
    }
    itsOutputImg = tmp;

    evolveBrain();
}


void BiasImageForm::showLabels( bool show )
{
    itsImg = itsTestScenes->getScene(itsCurrentScene); //load scene 0 by default
    itsSceneNumber->setMaxValue(itsTestScenes->getNumScenes());
    Image<PixRGB<byte> > tmp = itsImg;
    if (viewShow_LabelsAction->isOn())
      itsTestScenes->labelScene(itsCurrentScene, tmp);
    imgDisp->setImage(tmp);

}


void BiasImageForm::biasForObject( int obj )
{
  LFATAL("fixme");
  /*
  TestImages::ObjData objData = itsTestScenes->getObjectData(itsCurrentScene, obj, true);
  LINFO("Biasing for %i %s\n", obj, objData.description.c_str());
  Image<float> objMask = itsTestScenes->getObjMask(itsCurrentScene, obj, itsImg.getDims());
  MaskBiaser mb(objMask, true);
  ComplexChannel* vc = &*dynCastWeak<ComplexChannel>(itsBrain->getVC());
  vc->accept(mb);
  setSubmapAlgorithmBiased(*vc);
  showObjectLabel(true);
  */
}


void BiasImageForm::showObjectLabel( bool show )
{
  int lineWidth = int(itsImg.getWidth()*0.005);

  itsImg = itsTestScenes->getScene(itsCurrentScene); //load scene 0 by default
  Image<PixRGB<byte> > tmp = itsImg;
  if (viewShow_Object_LabelAction->isOn())
  {
    TestImages::ObjData objData = itsTestScenes->getObjectData(itsCurrentScene, itsSelectObject->currentItem());
    std::vector<Point2D<int> > objPoly = objData.polygon;
    Point2D<int> p1 = objPoly[0];
    for(uint i=1; i<objPoly.size(); i++)
    {
      drawLine(tmp, p1, objPoly[i], PixRGB<byte>(255, 0, 0), lineWidth);
      p1 = objPoly[i];
    }
    drawLine(tmp, p1, objPoly[0], PixRGB<byte>(255, 0, 0), lineWidth); //close the polygon

  }
  imgDisp->setImage(tmp);

}

void BiasImageForm::logFixation(const char *name, const int x, const int y, const std::vector<double> &FV)
{
  FILE *fp = fopen("fixation.log", "a");

  TestImages::SceneData sceneData = itsTestScenes->getSceneData(itsCurrentScene);

  fprintf(fp, "T: %s %s %i %i ", sceneData.filename.c_str(), name, x, y);
  for(uint i=0; i<FV.size(); i++)
  {
    fprintf(fp, "%f ", FV[i]);
    printf( "%f ", FV[i]);
  }
  fprintf(fp, "\n");
  printf("\n");

  fclose(fp);
}
