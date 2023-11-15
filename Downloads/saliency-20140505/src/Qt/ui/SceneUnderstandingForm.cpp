/****************************************************************************
** Form implementation generated from reading ui file 'Qt/SceneUnderstandingForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/SceneUnderstandingForm.h"

#include <qvariant.h>
#include <qfiledialog.h>
#include <qlineedit.h>
#include <qimage.h>
#include <qpixmap.h>
#include <qmessagebox.h>
#include <qpushbutton.h>
#include <qtabwidget.h>
#include <qwidget.h>
#include <qlabel.h>
#include <qspinbox.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>
#include <qimage.h>
#include <qpixmap.h>

#include "Raster/Raster.H"
#include "QtUtil/ImageConvert.H"
#include "SIFT/Histogram.H"
#include "Image/FilterOps.H"
#include "Image/ShapeOps.H"
#include "Qt/ImageCanvas.h"
#include "Qt/SceneUnderstandingForm.ui.h"
static const unsigned char image1_data[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x16,
    0x08, 0x06, 0x00, 0x00, 0x00, 0xc4, 0xb4, 0x6c, 0x3b, 0x00, 0x00, 0x00,
    0x99, 0x49, 0x44, 0x41, 0x54, 0x38, 0x8d, 0xed, 0x94, 0x41, 0x0e, 0x85,
    0x20, 0x0c, 0x44, 0x5f, 0x89, 0xc7, 0x36, 0x7f, 0x61, 0xbc, 0x77, 0x5d,
    0x28, 0x48, 0xa4, 0x28, 0x60, 0xff, 0xce, 0xd9, 0x54, 0x8b, 0xbe, 0x8e,
    0x13, 0x04, 0x3e, 0x1d, 0x92, 0x81, 0x77, 0xf4, 0x81, 0xa1, 0x23, 0xdc,
    0x2b, 0x34, 0xf6, 0xf4, 0x7a, 0x3d, 0xe2, 0xb8, 0x65, 0xa8, 0x84, 0x3f,
    0x40, 0x01, 0x98, 0x2a, 0x0b, 0x3d, 0x5f, 0x62, 0xc5, 0x83, 0x00, 0xaa,
    0x1a, 0xd7, 0x05, 0x50, 0x44, 0x9a, 0xb9, 0xd5, 0x07, 0xa7, 0x73, 0xa8,
    0xa4, 0xba, 0x4f, 0x92, 0xa2, 0xdf, 0x33, 0x3c, 0x64, 0xc6, 0x3b, 0xeb,
    0xbd, 0x82, 0xe5, 0xb8, 0xad, 0xde, 0xcb, 0xcc, 0x78, 0x20, 0xeb, 0x42,
    0x66, 0xc6, 0x39, 0x74, 0x5d, 0xfa, 0x80, 0xf3, 0x6f, 0xaf, 0x66, 0xc6,
    0x6f, 0xa1, 0x9c, 0x3f, 0x88, 0x2f, 0xb4, 0x70, 0xec, 0x05, 0xcd, 0xc0,
    0xbe, 0xd0, 0x78, 0x93, 0xf6, 0x8e, 0x17, 0x14, 0x92, 0x63, 0x5f, 0x68,
    0x6c, 0x3e, 0xef, 0xf6, 0xba, 0x3c, 0x8f, 0xdd, 0x36, 0x6d, 0xc4, 0xc0,
    0x45, 0x2c, 0x87, 0x81, 0xf8, 0x08, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
    0x4e, 0x44, 0xae, 0x42, 0x60, 0x82
};


/*
 *  Constructs a SceneUnderstandingForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
SceneUnderstandingForm::SceneUnderstandingForm( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    QImage img;
    img.loadFromData( image1_data, sizeof( image1_data ), "PNG" );
    image1 = img;
    if ( !name )
        setName( "SceneUnderstandingForm" );
    setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)7, 0, 0, sizePolicy().hasHeightForWidth() ) );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    SceneUnderstandingFormLayout = new QVBoxLayout( centralWidget(), 11, 6, "SceneUnderstandingFormLayout");

    tabDisp = new QTabWidget( centralWidget(), "tabDisp" );

    tab = new QWidget( tabDisp, "tab" );
    tabLayout = new QHBoxLayout( tab, 11, 6, "tabLayout");

    imgDisp = new ImageCanvas( tab, "imgDisp" );
    tabLayout->addWidget( imgDisp );
    tabDisp->insertTab( tab, QString::fromLatin1("") );
    SceneUnderstandingFormLayout->addWidget( tabDisp );

    layout2 = new QHBoxLayout( 0, 0, 6, "layout2");
    spacer9 = new QSpacerItem( 70, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout2->addItem( spacer9 );

    EvolveBrainButton = new QPushButton( centralWidget(), "EvolveBrainButton" );
    layout2->addWidget( EvolveBrainButton );
    spacer3 = new QSpacerItem( 110, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout2->addItem( spacer3 );

    runButton = new QPushButton( centralWidget(), "runButton" );
    layout2->addWidget( runButton );

    textLabel1 = new QLabel( centralWidget(), "textLabel1" );
    layout2->addWidget( textLabel1 );

    timesSpinBox = new QSpinBox( centralWidget(), "timesSpinBox" );
    timesSpinBox->setMaxValue( 1000 );
    timesSpinBox->setMinValue( 0 );
    timesSpinBox->setValue( 1 );
    layout2->addWidget( timesSpinBox );
    spacer3_2 = new QSpacerItem( 161, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout2->addItem( spacer3_2 );

    GenScenepushButton = new QPushButton( centralWidget(), "GenScenepushButton" );
    layout2->addWidget( GenScenepushButton );
    SceneUnderstandingFormLayout->addLayout( layout2 );

    layout3 = new QHBoxLayout( 0, 0, 6, "layout3");

    textLabel1_2 = new QLabel( centralWidget(), "textLabel1_2" );
    layout3->addWidget( textLabel1_2 );

    dialogText = new QLineEdit( centralWidget(), "dialogText" );
    layout3->addWidget( dialogText );
    SceneUnderstandingFormLayout->addLayout( layout3 );

    layout6 = new QHBoxLayout( 0, 0, 6, "layout6");

    textLabel2 = new QLabel( centralWidget(), "textLabel2" );
    textLabel2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, textLabel2->sizePolicy().hasHeightForWidth() ) );
    layout6->addWidget( textLabel2 );

    msgLabel = new QLabel( centralWidget(), "msgLabel" );
    layout6->addWidget( msgLabel );
    SceneUnderstandingFormLayout->addLayout( layout6 );

    // actions
    fileOpenAction = new QAction( this, "fileOpenAction" );
    fileOpenAction->setIconSet( QIconSet( image1 ) );
    fileSaveAsAction = new QAction( this, "fileSaveAsAction" );
    fileExitAction = new QAction( this, "fileExitAction" );
    editConfigureAction = new QAction( this, "editConfigureAction" );
    editConfigureAction->setToggleAction( TRUE );
    editConfigureAction->setOn( FALSE );
    editBias_SettingsAction = new QAction( this, "editBias_SettingsAction" );
    viewActionGroup = new QActionGroup( this, "viewActionGroup" );
    viewActionGroup->setOn( FALSE );
    viewActionGroup->setExclusive( FALSE );
    viewActionGroup->setUsesDropDown( FALSE );
    viewTrajAction = new QAction( viewActionGroup, "viewTrajAction" );
    viewTrajAction->setToggleAction( TRUE );
    viewSMapAction = new QAction( viewActionGroup, "viewSMapAction" );
    viewSMapAction->setToggleAction( TRUE );
    viewChannelsAction = new QAction( viewActionGroup, "viewChannelsAction" );
    viewChannelsAction->setToggleAction( TRUE );
    editDescriptor_VecAction = new QAction( this, "editDescriptor_VecAction" );
    editBayes_NetworkAction = new QAction( this, "editBayes_NetworkAction" );
    editBayes_NetworkViewAction = new QAction( this, "editBayes_NetworkViewAction" );
    editBayes_NetworkLoad_NetworkAction = new QAction( this, "editBayes_NetworkLoad_NetworkAction" );
    editBayes_NetworkSave_NetworkAction = new QAction( this, "editBayes_NetworkSave_NetworkAction" );
    editConfigureBias_ImageAction = new QAction( this, "editConfigureBias_ImageAction" );
    editConfigureBias_ImageAction->setToggleAction( TRUE );
    editConfigureBias_ImageAction->setOn( FALSE );
    editConfigureScene_SettingsAction = new QAction( this, "editConfigureScene_SettingsAction" );
    editConfigureTrainAction = new QAction( this, "editConfigureTrainAction" );
    editConfigureTrainAction->setToggleAction( TRUE );
    editConfigureTestAction = new QAction( this, "editConfigureTestAction" );
    editConfigureTestAction->setToggleAction( TRUE );
    fileOpen_WorkspaceAction = new QAction( this, "fileOpen_WorkspaceAction" );


    // toolbars


    // menubar
    MenuBar = new QMenuBar( this, "MenuBar" );


    fileMenu = new QPopupMenu( this );
    fileOpenAction->addTo( fileMenu );
    fileOpen_WorkspaceAction->addTo( fileMenu );
    fileSaveAsAction->addTo( fileMenu );
    fileMenu->insertSeparator();
    fileMenu->insertSeparator();
    fileExitAction->addTo( fileMenu );
    MenuBar->insertItem( QString(""), fileMenu, 1 );

    Edit = new QPopupMenu( this );
    popupMenu_6 = new QPopupMenu( this );
    Edit->insertItem( editConfigureAction->iconSet(), tr( "&Configure" ), popupMenu_6 );
    editConfigureBias_ImageAction->addTo( popupMenu_6 );
    editConfigureScene_SettingsAction->addTo( popupMenu_6 );
    editConfigureTestAction->addTo( popupMenu_6 );
    editBias_SettingsAction->addTo( Edit );
    editDescriptor_VecAction->addTo( Edit );
    popupMenu_12 = new QPopupMenu( this );
    Edit->insertItem( editBayes_NetworkAction->iconSet(), tr( "&Bayes Network" ), popupMenu_12 );
    editBayes_NetworkViewAction->addTo( popupMenu_12 );
    editBayes_NetworkLoad_NetworkAction->addTo( popupMenu_12 );
    editBayes_NetworkSave_NetworkAction->addTo( popupMenu_12 );
    MenuBar->insertItem( QString(""), Edit, 2 );

    View = new QPopupMenu( this );
    viewChannelsAction->addTo( View );
    viewSMapAction->addTo( View );
    viewTrajAction->addTo( View );
    MenuBar->insertItem( QString(""), View, 3 );

    languageChange();
    resize( QSize(763, 784).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( fileOpenAction, SIGNAL( activated() ), this, SLOT( fileOpen() ) );
    connect( fileExitAction, SIGNAL( activated() ), this, SLOT( close() ) );
    connect( editBias_SettingsAction, SIGNAL( activated() ), this, SLOT( showBiasSettings() ) );
    connect( viewActionGroup, SIGNAL( selected(QAction*) ), this, SLOT( configureView(QAction*) ) );
    connect( imgDisp, SIGNAL( mousePressed(int,int) ), this, SLOT( getDescriptor(int,int) ) );
    connect( editDescriptor_VecAction, SIGNAL( activated() ), this, SLOT( showDescriptorVec() ) );
    connect( GenScenepushButton, SIGNAL( clicked() ), this, SLOT( genScene() ) );
    connect( runButton, SIGNAL( clicked() ), this, SLOT( run() ) );
    connect( editBayes_NetworkLoad_NetworkAction, SIGNAL( activated() ), this, SLOT( loadBayesNetwork() ) );
    connect( editBayes_NetworkSave_NetworkAction, SIGNAL( activated() ), this, SLOT( saveBayesNetwork() ) );
    connect( editBayes_NetworkViewAction, SIGNAL( activated() ), this, SLOT( viewBayesNetwork() ) );
    connect( editConfigureBias_ImageAction, SIGNAL( toggled(bool) ), this, SLOT( setBiasImage(bool) ) );
    connect( EvolveBrainButton, SIGNAL( pressed() ), this, SLOT( evolveBrain() ) );
    connect( editConfigureScene_SettingsAction, SIGNAL( activated() ), this, SLOT( showSceneSettings() ) );
    connect( dialogText, SIGNAL( returnPressed() ), this, SLOT( submitDialog() ) );
    connect( fileOpen_WorkspaceAction, SIGNAL( activated() ), this, SLOT( fileOpenWorkspace() ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
SceneUnderstandingForm::~SceneUnderstandingForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void SceneUnderstandingForm::languageChange()
{
    setCaption( tr( "Bias Image" ) );
    tabDisp->changeTab( tab, tr( "Original" ) );
    EvolveBrainButton->setText( tr( "Evolve Brain" ) );
    runButton->setText( tr( "Run" ) );
    textLabel1->setText( tr( "Epochs:" ) );
    GenScenepushButton->setText( tr( "Generate\n"
"Scene" ) );
    textLabel1_2->setText( tr( "Dialog:" ) );
    textLabel2->setText( tr( "Mesg:" ) );
    msgLabel->setText( QString::null );
    fileOpenAction->setText( tr( "Open" ) );
    fileOpenAction->setMenuText( tr( "&Open..." ) );
    fileOpenAction->setAccel( tr( "Ctrl+O" ) );
    fileSaveAsAction->setText( tr( "Save As" ) );
    fileSaveAsAction->setMenuText( tr( "Save &As..." ) );
    fileSaveAsAction->setAccel( QString::null );
    fileExitAction->setText( tr( "Exit" ) );
    fileExitAction->setMenuText( tr( "E&xit" ) );
    fileExitAction->setAccel( QString::null );
    editConfigureAction->setText( tr( "Configure" ) );
    editConfigureAction->setMenuText( tr( "&Configure" ) );
    editBias_SettingsAction->setText( tr( "&Bias Settings" ) );
    editBias_SettingsAction->setMenuText( tr( "&Bias Settings" ) );
    viewActionGroup->setText( tr( "View" ) );
    viewActionGroup->setAccel( tr( "Alt+F, S" ) );
    viewTrajAction->setText( tr( "View Traj" ) );
    viewTrajAction->setMenuText( tr( "View &Traj" ) );
    viewTrajAction->setAccel( tr( "Ctrl+T" ) );
    viewSMapAction->setText( tr( "View SMap" ) );
    viewSMapAction->setAccel( tr( "Ctrl+S" ) );
    viewChannelsAction->setText( tr( "view Channels" ) );
    editDescriptor_VecAction->setText( tr( "&Descriptor Vec" ) );
    editDescriptor_VecAction->setMenuText( tr( "&Descriptor Vec" ) );
    editBayes_NetworkAction->setText( tr( "&Bayes Network" ) );
    editBayes_NetworkAction->setMenuText( tr( "&Bayes Network" ) );
    editBayes_NetworkViewAction->setText( tr( "View" ) );
    editBayes_NetworkViewAction->setMenuText( tr( "View" ) );
    editBayes_NetworkLoad_NetworkAction->setText( tr( "Load Network" ) );
    editBayes_NetworkLoad_NetworkAction->setMenuText( tr( "Load Network" ) );
    editBayes_NetworkSave_NetworkAction->setText( tr( "Save Network" ) );
    editBayes_NetworkSave_NetworkAction->setMenuText( tr( "Save Network" ) );
    editConfigureBias_ImageAction->setText( tr( "Bias Image" ) );
    editConfigureBias_ImageAction->setMenuText( tr( "Bias Image" ) );
    editConfigureScene_SettingsAction->setText( tr( "Scene Settings" ) );
    editConfigureScene_SettingsAction->setMenuText( tr( "Scene Settings" ) );
    editConfigureTrainAction->setText( tr( "Train" ) );
    editConfigureTrainAction->setMenuText( tr( "Train" ) );
    editConfigureTestAction->setText( tr( "Test" ) );
    editConfigureTestAction->setMenuText( tr( "Test" ) );
    fileOpen_WorkspaceAction->setText( tr( "Open &Workspace" ) );
    fileOpen_WorkspaceAction->setMenuText( tr( "Open &Workspace" ) );
    if (MenuBar->findItem(1))
        MenuBar->findItem(1)->setText( tr( "&File" ) );
    Edit->changeItem( Edit->idAt( 0 ), tr( "&Configure" ) );
    Edit->changeItem( Edit->idAt( 3 ), tr( "&Bayes Network" ) );
    if (MenuBar->findItem(2))
        MenuBar->findItem(2)->setText( tr( "&Edit" ) );
    if (MenuBar->findItem(3))
        MenuBar->findItem(3)->setText( tr( "&View" ) );
}

