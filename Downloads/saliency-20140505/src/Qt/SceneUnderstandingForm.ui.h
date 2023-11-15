/*! @file Qt/SceneUnderstandingForm.ui.h Main form for scene undersanding */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/SceneUnderstandingForm.ui.h $
// $Id: SceneUnderstandingForm.ui.h 9412 2008-03-10 23:10:15Z farhan $

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


#include "Channels/SubmapAlgorithmBiased.H"

void SceneUnderstandingForm::init( ModelManager & manager )
{

  itsMgr = &manager;

  //Init the descriptor vector
  nub::ref<StdBrain>  brain = dynCastWeak<StdBrain>(itsMgr->subComponent("Brain"));


  //The SceneUnderstanding
  itsSceneUnderstanding = new SceneUnderstanding(itsMgr, brain);

  itsDescriptorVec = itsSceneUnderstanding->getDescriptorVecPtr();
  itsBayesNetwork = itsSceneUnderstanding->getBayesPtr();
  itsProlog = itsSceneUnderstanding->getPrologPtr();

  itsDescriptorVecDialog.init(*itsDescriptorVec);


  itsTrainScene = new SceneGenerator(SceneGenerator::ALOI_OBJECTS, 500, 500);
  itsTestScene = new SceneGenerator(SceneGenerator::ALOI_OBJECTS, 500, 500);

}

void SceneUnderstandingForm::fileOpen()
{

  static QString prevFilename("/home/lior/scenes/satellite/images/set1");

  QString file = QFileDialog::getOpenFileName( prevFilename,
      "Images (*.pgm *.ppm *.pnm *.jpeg *.png *.jpg)",
      this, "OpenImageDialog",
      "Choose Image" );
  if( !file.isEmpty() ) {
    prevFilename = file;
    itsImg = Raster::ReadRGB( file );
    printf("File: %s \n", file.ascii());
   // itsImg = rescale(itsImg, 680, 512);
    itsSceneUnderstanding->setImage(itsImg);
    evolveBrain();

   /* //draw ref circles
    for(int y=0; y<itsImg.getHeight()+70; y+=70)
      for(int x=0; x<itsImg.getWidth()+80; x+=80)
      {
        if ((y/70)%2)
          drawCircle(itsImg, Point2D<int>(x-40,y), 40, PixRGB<byte>(255,0,0));
        else
          drawCircle(itsImg, Point2D<int>(x,y), 40, PixRGB<byte>(255,0,0));
      }*/
    imgDisp->setImage(itsImg);
    updateDisplay();
  }
}

void SceneUnderstandingForm::fileOpenWorkspace()
{

  QString KBFile("/home/lior/scenes/satellite/scenes/scene");
  QString bayesFile("/home/lior/scenes/satellite/scenes/scene.net");

  LINFO("Consulting with %s", KBFile.ascii());
  if( itsProlog->consult((char*)KBFile.ascii()) )
  {
    msgLabel->setText(QString("KB Loaded: ") + KBFile);
  } else {
    msgLabel->setText(QString("Can not load KB: ") + KBFile);
  }

  itsBayesNetwork->load(bayesFile.ascii());

}

void SceneUnderstandingForm::updateDisplay()
{
    nub::ref<StdBrain>  brain = itsSceneUnderstanding->getBrainPtr();
    if (true) showTraj(brain);
    if (true) showSMap(brain);
    if (true) showChannels(brain);

}

void SceneUnderstandingForm::fileSave()
{

}


void SceneUnderstandingForm::fileExit()
{
}


void SceneUnderstandingForm::showBiasSettings()
{
  itsBiasSettingsDialog.init(*itsMgr);
  itsBiasSettingsDialog.show();

}


void SceneUnderstandingForm::evolveBrain()
{

  float interestLevel = itsSceneUnderstanding->evolveBrain();
  updateDisplay();
  LINFO("Interest Level %f", interestLevel);
  if (interestLevel > 1.0F)
  {
    msgLabel->setText(QString("Found something interesting %L1").arg(interestLevel));
    Point2D<int> foveaLoc = itsSceneUnderstanding->getFoveaLoc();
    classifyFovea(foveaLoc.i, foveaLoc.j);
  }  else {
    msgLabel->setText(QString("Boring"));
  }

}

void SceneUnderstandingForm::showTraj(nub::ref<StdBrain>& brain){
  static bool setupTab = true;
  static ImageCanvas* disp = NULL;

  //get the Seq com
  nub::ref<SimEventQueueConfigurator> seqc =
    dynCastWeak<SimEventQueueConfigurator>(itsMgr->subComponent("SimEventQueueConfigurator"));
  nub::ref<SimEventQueue> itsSEQ = seqc->getQ();

  itsOutputImg = brain->getSV()->getTraj(itsSEQ->now());

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


void SceneUnderstandingForm::showSMap( nub::ref<StdBrain> & brain )
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
    Image<float> img = brain->getSM()->getV(false);
    disp->setImage(img);
  }

}


void SceneUnderstandingForm::showChannels( nub::ref<StdBrain> & brain )
{
  static bool setupTab = true;
  static ImageCanvas* disp[10];
  int numChans = brain->getVC()->numChans();

  if (setupTab){
    for (int i=0; i<numChans; i++)
    {
      QWidget* tab = new QWidget(tabDisp);
      QHBoxLayout* tabLayout = new QHBoxLayout(tab, 11, 6);
      disp[i] = new ImageCanvas(tab);
      tabLayout->addWidget(disp[i]);
      tabDisp->insertTab(tab,
          brain->getVC()->subChan(i)->descriptiveName());
      setupTab = false;
    }
  }

  if (disp)
  {
    for( int i=0; i<numChans; i++){
      Image<float> img = brain->getVC()->subChan(i)->getOutput();
      disp[i]->setImage(img);
    }
  }

}


void SceneUnderstandingForm::configureView( QAction * action )
{
  LINFO("Configure view %i %i %i", itsViewTraj, itsViewSMap, itsViewChannels);
  if (viewTrajAction->isOn()) itsViewTraj = true; else itsViewTraj = false;
  if (viewSMapAction->isOn()) itsViewSMap = true; else itsViewSMap = false;
  if (viewChannelsAction->isOn()) itsViewChannels = true; else itsViewChannels = false;
}


void SceneUnderstandingForm::setBias( int x, int y )
{

}


void SceneUnderstandingForm::getDescriptor( int x, int y )
{
  printf("%i,%i\n", x, y);
  nub::ref<StdBrain>  brain = dynCastWeak<StdBrain>(itsMgr->subComponent("Brain"));
  ComplexChannel* cc = &*dynCastWeak<ComplexChannel>(brain->getVC());
  itsCurrentAttention = Point2D<int>(x,y);
  //check if the brain has evolved
  if (!cc->hasInput())
    evolveBrain();

  //Classify the object under the fovea
  if (cc->hasInput()) //check if we got an input (could be a blank image)
  {
    classifyFovea(x, y);
  }
}




void SceneUnderstandingForm::showDescriptorVec()
{
  itsDescriptorVecDialog.show();
}


void SceneUnderstandingForm::genScene()
{

  if (!editConfigureTestAction->isOn())  //Train Mode
    itsImg = itsTrainScene->getScene(9);
  else
    itsImg = itsTestScene->getScene(9);

  imgDisp->setImage(itsImg);
  //   evolveBrain();

}

void SceneUnderstandingForm::classifyFovea(int x, int y)
{

  double prob;
  int cls = itsSceneUnderstanding->classifyFovea(Point2D<int>(x,y), &prob);
  itsDescriptorVecDialog.update();

  if (cls == -1)
  {
    msgLabel->setText(QString("Can you tell me what this is?"));
    itsCurrentObject = -1;
  } else {
    LINFO("Class name is %s (%f)", itsBayesNetwork->getClassName(cls), prob);
    msgLabel->setText(QString("This is %L1 (%L2)")
        .arg(itsBayesNetwork->getClassName(cls))
        .arg(prob));
    itsCurrentObject = cls;
  }


}




void SceneUnderstandingForm::run()
{
  for (int i=0; i<timesSpinBox->value(); i++){
    genScene(); //generate a scene
    evolveBrain(); //evolve the brain
  }

}


void SceneUnderstandingForm::loadBayesNetwork()
{
  QString file = QFileDialog::getOpenFileName( QString::null,
      "Bayes network (*.net)",
      this, "OpenImageDialog",
      "Choose Bayes Network" );
  if( !file.isEmpty() )
    itsBayesNetwork->load(file.ascii());

}


void SceneUnderstandingForm::saveBayesNetwork()
{
  QString file = QFileDialog::getSaveFileName( QString::null,
      "Bayes network (*.net)",
      this, "SaveImageDialog",
      "Choose Bayes Network" );
  if( !file.isEmpty() )
    itsBayesNetwork->save(file.ascii());


}


void SceneUnderstandingForm::viewBayesNetwork()
{
  itsBayesNetworkDialog.show();
  itsBayesNetworkDialog.init(*itsBayesNetwork);

}


void SceneUnderstandingForm::setBiasImage( bool biasVal )
{
  nub::ref<StdBrain>  brain = dynCastWeak<StdBrain>(itsMgr->subComponent("Brain"));
  ComplexChannel* cc = &*dynCastWeak<ComplexChannel>(brain->getVC());

  //set the bias
  setSubmapAlgorithmBiased(*cc);
}


void SceneUnderstandingForm::showSceneSettings()
{
  itsSceneSettingsDialog.init(itsTrainScene, itsTestScene);
  itsSceneSettingsDialog.show();

}


void SceneUnderstandingForm::submitDialog()
{

  QString str = dialogText->text().lower();
  LINFO("You said: %s", str.ascii());

  if (str.startsWith("this is ")) //bias for the object
  {
    QString obj = str.section(' ', 2, 2);
    msgLabel->setText(QString("Learning ") + obj);

    //Learn the object
    LINFO("Learning object from %ix%i", itsCurrentAttention.i, itsCurrentAttention.j);

    itsSceneUnderstanding->learn(itsCurrentAttention, obj.ascii());
    itsDescriptorVecDialog.update();
    updateDisplay();

    return;
  }

  if (str.startsWith("where is ")) //bias for the object
  {

    QString obj = str.section(' ', 2, 2);
    bool objKnown = itsSceneUnderstanding->biasFor(obj.ascii());

    if (objKnown ) //we know about this object
    {
      msgLabel->setText(QString("Finding ") + obj);
      LINFO("Biasing for %s", obj.ascii());
      evolveBrain();
    } else {
      msgLabel->setText(QString("I am sorry, but I dont know anything about ") + obj);
    }
    updateDisplay();
    return ;

  }

  //prolog interface
  if (str.startsWith("consult ")) //load a prolog file
  {
    QString KBFile = str.section(' ', 1, 1);
    msgLabel->setText(QString("Loading KB: ") + KBFile);

                LINFO("Consulting with %s", KBFile.ascii());
    if( itsProlog->consult((char*)KBFile.ascii()) )
    {
                        msgLabel->setText(QString("KB Loaded: ") + KBFile);
                } else {
                        msgLabel->setText(QString("Can not load KB: ") + KBFile);
                }

                return;
        }


  if (str.startsWith("go")) //load a prolog file
        {

    std::string sceneType = itsSceneUnderstanding->highOrderRec();
    updateDisplay();

    QString msg;
    msg = QString("Scene is: ") + sceneType;
    msgLabel->setText(msg);

                return;

  }

  msgLabel->setText(QString("I did not understand!"));


}




