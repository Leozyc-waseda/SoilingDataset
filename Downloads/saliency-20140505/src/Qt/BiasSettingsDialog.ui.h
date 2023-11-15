/*! @file Qt/BiasSettingsDialog.ui.h Dialog for biasing images */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/BiasSettingsDialog.ui.h $
// $Id: BiasSettingsDialog.ui.h 10827 2009-02-11 09:40:02Z itti $


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


void BiasSettingsDialog::init( ModelManager & manager )
{
  static bool isInit = false;

  if (!isInit){

    //remove the first dummy page
    QWidget *page = tabDisp->page(0); //get the firs page of the tab
    tabDisp->removePage(page);

    LFATAL("fixme");
    /*
    itsMgr = &manager;
    //        showFeatures();
    nub::ref<StdBrain>  brain = dynCastWeak<StdBrain>(itsMgr->subComponent("Brain"));

    for (unsigned int i=0; i<brain->getVC()->numChans(); i++){
      nub::ref<ChannelBase> cb = brain->getVC()->subChan(i);

      ComplexChannel* const cc = dynamic_cast<ComplexChannel*>(cb.get());

      if (cc != 0){ //do we have a complex channel or a simple one
        const int numc = cc->numChans();
        LINFO("Channel %s nc:%i", cb->descriptiveName().c_str(), numc);
        for (int j=0; j<numc; j++){
          nub::ref<ChannelBase> cb = cc->subChan(j);
          SingleChannel& sc = dynamic_cast<SingleChannel&>(*cb);
          setupTab(*cc, sc);
        }
      } else {
        LINFO("Channel %s nc:%i", cb->descriptiveName().c_str(), 0);
        SingleChannel& sc = dynamic_cast<SingleChannel&>(*cb);
        setupTab(*(brain->getVC()), sc);
      }
    }
    isInit = true;
    */
  }
}


void BiasSettingsDialog::biasFeature(int value)
{

}

void BiasSettingsDialog::showFeatures( )
{

}


void BiasSettingsDialog::setupTab( ComplexChannel& cc, SingleChannel &sc )
{
  int nsubmap = sc.numSubmaps();
  QWidget* tab = new QWidget(tabDisp);
  QGridLayout* tabLayout = new QGridLayout( tab, 1, 1, 3, 3, "tabLayout");


  /*ImageCanvas* combinedDisp = new ImageCanvas(tab);
    Image<float> img = sc.getOutput();
    inplaceNormalize(img, 0.0F, 255.0F);
    Image<PixRGB<byte> > colImg = toRGB(img);
    combinedDisp->setImage(colImg);*/

  BiasValImage *combinedDisp = new BiasValImage(cc, sc, -1, tab);

  for(int i=0; i<nsubmap; i++){
    int xpos = (i/2);
    int ypos = i%2;
    unsigned int clev = 0, slev = 0;
    sc.getLevelSpec().indexToCS(i, clev, slev);

    BiasValImage *biasValImage = new BiasValImage(cc, sc, i, tab);
    itsBiasValImage.push_back(biasValImage);
    //signal the combined display on updates
    connect(biasValImage, SIGNAL(updateOutput()),
        combinedDisp, SLOT(updateValues()));

    //update the display when the update button is pressed
    connect(updateValButton, SIGNAL(clicked()),
        biasValImage, SLOT(updateValues()));

    tabLayout->addWidget( biasValImage, xpos, ypos );

  }

  tabLayout->addMultiCellWidget( combinedDisp, 1, 1, 2, 2 );
  tabDisp->insertTab( tab, sc.descriptiveName());

}


void BiasSettingsDialog::update()
{
  for(uint i=0; i<itsBiasValImage.size(); i++)
  {
    itsBiasValImage[i]->setShowRaw(chkBoxShowRaw->isChecked());
    itsBiasValImage[i]->setResizeToSLevel(chkBoxResizeToSLevel->isChecked());
  }

}
