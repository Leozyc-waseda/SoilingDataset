/****************************************************************************
** Form interface generated from reading ui file 'Qt/BiasImageForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef BIASIMAGEFORM_H
#define BIASIMAGEFORM_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qmainwindow.h>
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/DrawOps.H"
#include "Image/CutPaste.H"
#include "BiasSettingsDialog.h"
#include "Component/ModelManager.H"
#include "Neuro/Brain.H"
#include "Neuro/SimulationViewerStd.H"
#include "Neuro/VisualCortex.H"
#include "Qt/ImageCanvas.h"
#include "Neuro/SaliencyMap.H"
#include "Image/ColorOps.H"
#include "Media/TestImages.H"
#include "Channels/DescriptorVec.H"
#include "Learn/Bayes.H"
#include "DescriptorVecDialog.h"
#include "BayesNetworkDialog.h"
#include "SceneSettingsDialog.h"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QAction;
class QActionGroup;
class QToolBar;
class QPopupMenu;
class ImageCanvas;
class QTabWidget;
class QWidget;
class QLabel;
class QSpinBox;
class QComboBox;
class QPushButton;
class QLineEdit;

class BiasImageForm : public QMainWindow
{
    Q_OBJECT

public:
    BiasImageForm( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~BiasImageForm();

    QTabWidget* tabDisp;
    QWidget* tab;
    ImageCanvas* imgDisp;
    QLabel* textLabel1_2;
    QSpinBox* itsSceneNumber;
    QComboBox* itsSelectObject;
    QPushButton* EvolveBrainButton;
    QPushButton* runButton;
    QLabel* textLabel1;
    QSpinBox* timesSpinBox;
    QLabel* textLabel2;
    QLabel* msgLabel;
    QLineEdit* itsObjectName;
    QMenuBar *MenuBar;
    QPopupMenu *fileMenu;
    QPopupMenu *Edit;
    QPopupMenu *popupMenu_6;
    QPopupMenu *popupMenu_12;
    QPopupMenu *View;
    QAction* fileOpenAction;
    QAction* fileSaveAsAction;
    QAction* fileExitAction;
    QAction* editConfigureAction;
    QAction* editBias_SettingsAction;
    QActionGroup* viewActionGroup;
    QAction* viewTrajAction;
    QAction* viewSMapAction;
    QAction* viewChannelsAction;
    QAction* editDescriptor_VecAction;
    QAction* editBayes_NetworkAction;
    QAction* editBayes_NetworkViewAction;
    QAction* editBayes_NetworkLoad_NetworkAction;
    QAction* editBayes_NetworkSave_NetworkAction;
    QAction* editConfigureBias_ImageAction;
    QAction* editConfigureScene_SettingsAction;
    QAction* editConfigureTrainAction;
    QAction* editConfigureTestAction;
    QAction* viewShow_LabelsAction;
    QAction* viewShow_Object_LabelAction;

    virtual void init( ModelManager & manager, nub::ref<Brain> brain, nub::ref<SimEventQueue> seq );
    virtual void showTraj();
    virtual void showSMap();
    virtual void showChannels();
    virtual void classifyFovea( int x, int y, int button );
    virtual void logFixation(const char *name, const int x,const  int y,const  std::vector<double> & FV);

public slots:
    virtual void fileOpen();
    virtual void fileSave();
    virtual void fileExit();
    virtual void showBiasSettings();
    virtual void evolveBrain();
    virtual void configureView( QAction * action );
    virtual void getDescriptor( int x, int y, int button );
    virtual void showDescriptorVec();
    virtual void run();
    virtual void loadBayesNetwork();
    virtual void saveBayesNetwork();
    virtual void viewBayesNetwork();
    virtual void setBiasImage( bool biasVal );
    virtual void showSceneSettings();
    virtual void getScene( int scene );
    virtual void showLabels( bool show );
    virtual void biasForObject( int obj );
    virtual void showObjectLabel( bool show );

protected:
    int clickClass;
    TestImages *itsTestScenes;
    TestImages *itsTrainScenes;
    Bayes *itsBayesNetwork;
    DescriptorVecDialog itsDescriptorVecDialog;
    nub::soft_ref<SimEventQueue> itsSEQ;
    nub::soft_ref<Brain> itsBrain;
    ModelManager *itsMgr;
    Image< PixRGB<byte> > itsImg;
    BiasSettingsDialog itsBiasSettingsDialog;
    Image< PixRGB<byte> > itsOutputImg;
    DescriptorVec* itsDescriptorVec;
    std::vector<Point2D<int> > itsTargetsLoc;
    SceneSettingsDialog itsSceneSettingsDialog;
    BayesNetworkDialog itsBayesNetworkDialog;
    int itsCurrentScene;
    Point2D<int> itsCurrentWinner;
    bool itsNewImage;

    QVBoxLayout* BiasImageFormLayout;
    QHBoxLayout* tabLayout;
    QHBoxLayout* layout4;
    QSpacerItem* spacer3_2;
    QHBoxLayout* layout3;

protected slots:
    virtual void languageChange();

private:
    bool itsViewChannels;
    bool itsViewSMap;
    bool itsViewTraj;

    QPixmap image0;
    QPixmap image1;

};

#endif // BIASIMAGEFORM_H
