/****************************************************************************
** Form interface generated from reading ui file 'Qt/SceneUnderstandingForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef SCENEUNDERSTANDINGFORM_H
#define SCENEUNDERSTANDINGFORM_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qmainwindow.h>
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/DrawOps.H"
#include "Image/CutPaste.H"
#include "BiasSettingsDialog.h"
#include "Component/ModelManager.H"
#include "Neuro/StdBrain.H"
#include "Neuro/SimulationViewerStd.H"
#include "Neuro/VisualCortex.H"
#include "Qt/ImageCanvas.h"
#include "Neuro/SaliencyMap.H"
#include "Image/ColorOps.H"
#include "Media/SceneGenerator.H"
#include "Channels/DescriptorVec.H"
#include "Learn/Bayes.H"
#include "Learn/SWIProlog.H"
#include "DescriptorVecDialog.h"
#include "BayesNetworkDialog.h"
#include "SceneSettingsDialog.h"
#include "SceneUnderstanding/SceneUnderstanding.H"

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
class QPushButton;
class QLabel;
class QSpinBox;
class QLineEdit;

class SceneUnderstandingForm : public QMainWindow
{
    Q_OBJECT

public:
    SceneUnderstandingForm( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~SceneUnderstandingForm();

    QTabWidget* tabDisp;
    QWidget* tab;
    ImageCanvas* imgDisp;
    QPushButton* EvolveBrainButton;
    QPushButton* runButton;
    QLabel* textLabel1;
    QSpinBox* timesSpinBox;
    QPushButton* GenScenepushButton;
    QLabel* textLabel1_2;
    QLineEdit* dialogText;
    QLabel* textLabel2;
    QLabel* msgLabel;
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
    QAction* fileOpen_WorkspaceAction;

    virtual void init( ModelManager & manager );
    virtual void updateDisplay();
    virtual void showTraj( nub::ref<StdBrain> & brain );
    virtual void showSMap( nub::ref<StdBrain> & brain );
    virtual void showChannels( nub::ref<StdBrain> & brain );
    virtual void classifyFovea( int x, int y );

public slots:
    virtual void fileOpen();
    virtual void fileSave();
    virtual void fileExit();
    virtual void showBiasSettings();
    virtual void evolveBrain();
    virtual void configureView( QAction * action );
    virtual void setBias( int x, int y );
    virtual void getDescriptor( int x, int y );
    virtual void showDescriptorVec();
    virtual void genScene();
    virtual void run();
    virtual void loadBayesNetwork();
    virtual void saveBayesNetwork();
    virtual void viewBayesNetwork();
    virtual void setBiasImage( bool biasVal );
    virtual void showSceneSettings();
    virtual void submitDialog();
    virtual void fileOpenWorkspace();

protected:
    BayesNetworkDialog itsBayesNetworkDialog;
    SceneSettingsDialog itsSceneSettingsDialog;
    std::vector<Point2D> itsTargetsLoc;
    DescriptorVec* itsDescriptorVec;
    Image< PixRGB<byte> > itsOutputImg;
    BiasSettingsDialog itsBiasSettingsDialog;
    Image< PixRGB<byte> > itsImg;
    ModelManager *itsMgr;
    DescriptorVecDialog itsDescriptorVecDialog;
    Bayes *itsBayesNetwork;
    SWIProlog *itsProlog;
    SceneGenerator *itsTrainScene;
    SceneGenerator *itsTestScene;
    Point2D itsCurrentAttention;
    int itsCurrentObject;
    SceneUnderstanding *itsSceneUnderstanding;

    QVBoxLayout* SceneUnderstandingFormLayout;
    QHBoxLayout* tabLayout;
    QHBoxLayout* layout2;
    QSpacerItem* spacer9;
    QSpacerItem* spacer3;
    QSpacerItem* spacer3_2;
    QHBoxLayout* layout3;
    QHBoxLayout* layout6;

protected slots:
    virtual void languageChange();

private:
    bool itsViewTraj;
    bool itsViewSMap;
    bool itsViewChannels;

    QPixmap image0;
    QPixmap image1;

};

#endif // SCENEUNDERSTANDINGFORM_H
