/****************************************************************************
** Form implementation generated from reading ui file 'Qt/SceneSettingsDialog.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/SceneSettingsDialog.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qgroupbox.h>
#include <qlineedit.h>
#include <qcombobox.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Qt/SceneSettingsDialog.ui.h"

/*
 *  Constructs a SceneSettingsDialog as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
SceneSettingsDialog::SceneSettingsDialog( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "SceneSettingsDialog" );
    setSizeGripEnabled( TRUE );
    SceneSettingsDialogLayout = new QGridLayout( this, 1, 1, 11, 6, "SceneSettingsDialogLayout");

    groupBox1 = new QGroupBox( this, "groupBox1" );
    groupBox1->setColumnLayout(0, Qt::Vertical );
    groupBox1->layout()->setSpacing( 6 );
    groupBox1->layout()->setMargin( 11 );
    groupBox1Layout = new QGridLayout( groupBox1->layout() );
    groupBox1Layout->setAlignment( Qt::AlignTop );

    itsTrainingTargetObject = new QLineEdit( groupBox1, "itsTrainingTargetObject" );

    groupBox1Layout->addMultiCellWidget( itsTrainingTargetObject, 1, 1, 1, 6 );

    itsTrainingSceneType = new QComboBox( FALSE, groupBox1, "itsTrainingSceneType" );

    groupBox1Layout->addMultiCellWidget( itsTrainingSceneType, 0, 0, 1, 6 );

    textLabel1 = new QLabel( groupBox1, "textLabel1" );

    groupBox1Layout->addWidget( textLabel1, 0, 0 );

    textLabel2 = new QLabel( groupBox1, "textLabel2" );

    groupBox1Layout->addWidget( textLabel2, 1, 0 );

    textLabel7 = new QLabel( groupBox1, "textLabel7" );

    groupBox1Layout->addWidget( textLabel7, 2, 0 );

    textLabel4 = new QLabel( groupBox1, "textLabel4" );

    groupBox1Layout->addWidget( textLabel4, 3, 0 );

    itsTrainingRotation = new QLineEdit( groupBox1, "itsTrainingRotation" );
    itsTrainingRotation->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingRotation->sizePolicy().hasHeightForWidth() ) );
    itsTrainingRotation->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingRotation->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingRotation, 6, 2 );

    textLabel5_2 = new QLabel( groupBox1, "textLabel5_2" );

    groupBox1Layout->addWidget( textLabel5_2, 6, 0 );

    textLabel5 = new QLabel( groupBox1, "textLabel5" );

    groupBox1Layout->addWidget( textLabel5, 7, 0 );

    itsTrainingNoise = new QLineEdit( groupBox1, "itsTrainingNoise" );
    itsTrainingNoise->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingNoise->sizePolicy().hasHeightForWidth() ) );
    itsTrainingNoise->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingNoise->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingNoise, 7, 2 );

    textLabel6 = new QLabel( groupBox1, "textLabel6" );

    groupBox1Layout->addWidget( textLabel6, 4, 0 );

    textLabel3 = new QLabel( groupBox1, "textLabel3" );

    groupBox1Layout->addWidget( textLabel3, 5, 0 );

    textLabel10 = new QLabel( groupBox1, "textLabel10" );

    groupBox1Layout->addWidget( textLabel10, 2, 5 );

    itsTrainingColorB = new QLineEdit( groupBox1, "itsTrainingColorB" );
    itsTrainingColorB->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingColorB->sizePolicy().hasHeightForWidth() ) );
    itsTrainingColorB->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingColorB->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingColorB, 3, 6 );

    textLabel10_2 = new QLabel( groupBox1, "textLabel10_2" );

    groupBox1Layout->addWidget( textLabel10_2, 3, 5 );

    textLabel8_2 = new QLabel( groupBox1, "textLabel8_2" );

    groupBox1Layout->addWidget( textLabel8_2, 3, 1 );

    textLabel9_2_2 = new QLabel( groupBox1, "textLabel9_2_2" );

    groupBox1Layout->addWidget( textLabel9_2_2, 4, 3 );

    textLabel9_2 = new QLabel( groupBox1, "textLabel9_2" );

    groupBox1Layout->addWidget( textLabel9_2, 3, 3 );

    itsTrainingBackColorG = new QLineEdit( groupBox1, "itsTrainingBackColorG" );
    itsTrainingBackColorG->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingBackColorG->sizePolicy().hasHeightForWidth() ) );
    itsTrainingBackColorG->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingBackColorG->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingBackColorG, 4, 4 );

    textLabel9 = new QLabel( groupBox1, "textLabel9" );

    groupBox1Layout->addWidget( textLabel9, 2, 3 );

    itsTrainingTargetColorG = new QLineEdit( groupBox1, "itsTrainingTargetColorG" );
    itsTrainingTargetColorG->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingTargetColorG->sizePolicy().hasHeightForWidth() ) );
    itsTrainingTargetColorG->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingTargetColorG->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingTargetColorG, 2, 4 );

    itsTrainingColorG = new QLineEdit( groupBox1, "itsTrainingColorG" );
    itsTrainingColorG->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingColorG->sizePolicy().hasHeightForWidth() ) );
    itsTrainingColorG->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingColorG->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingColorG, 3, 4 );

    textLabel10_2_2 = new QLabel( groupBox1, "textLabel10_2_2" );

    groupBox1Layout->addWidget( textLabel10_2_2, 4, 5 );

    itsTrainingTargetColorB = new QLineEdit( groupBox1, "itsTrainingTargetColorB" );
    itsTrainingTargetColorB->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingTargetColorB->sizePolicy().hasHeightForWidth() ) );
    itsTrainingTargetColorB->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingTargetColorB->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingTargetColorB, 2, 6 );

    itsTrainingBackColorR = new QLineEdit( groupBox1, "itsTrainingBackColorR" );
    itsTrainingBackColorR->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingBackColorR->sizePolicy().hasHeightForWidth() ) );
    itsTrainingBackColorR->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingBackColorR->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingBackColorR, 4, 2 );

    textLabel8_2_2 = new QLabel( groupBox1, "textLabel8_2_2" );

    groupBox1Layout->addWidget( textLabel8_2_2, 4, 1 );

    itsTrainingBackColorB = new QLineEdit( groupBox1, "itsTrainingBackColorB" );
    itsTrainingBackColorB->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingBackColorB->sizePolicy().hasHeightForWidth() ) );
    itsTrainingBackColorB->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingBackColorB->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingBackColorB, 4, 6 );

    itsTrainingColorR = new QLineEdit( groupBox1, "itsTrainingColorR" );
    itsTrainingColorR->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingColorR->sizePolicy().hasHeightForWidth() ) );
    itsTrainingColorR->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingColorR->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingColorR, 3, 2 );

    textLabel8 = new QLabel( groupBox1, "textLabel8" );

    groupBox1Layout->addWidget( textLabel8, 2, 1 );

    itsTrainingLum = new QLineEdit( groupBox1, "itsTrainingLum" );
    itsTrainingLum->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingLum->sizePolicy().hasHeightForWidth() ) );
    itsTrainingLum->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingLum->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingLum, 5, 2 );

    itsTrainingTargetColorR = new QLineEdit( groupBox1, "itsTrainingTargetColorR" );
    itsTrainingTargetColorR->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTrainingTargetColorR->sizePolicy().hasHeightForWidth() ) );
    itsTrainingTargetColorR->setMaximumSize( QSize( 30, 23 ) );
    itsTrainingTargetColorR->setMaxLength( 3 );

    groupBox1Layout->addWidget( itsTrainingTargetColorR, 2, 2 );

    SceneSettingsDialogLayout->addWidget( groupBox1, 0, 0 );

    groupBox1_2 = new QGroupBox( this, "groupBox1_2" );
    groupBox1_2->setColumnLayout(0, Qt::Vertical );
    groupBox1_2->layout()->setSpacing( 6 );
    groupBox1_2->layout()->setMargin( 11 );
    groupBox1_2Layout = new QGridLayout( groupBox1_2->layout() );
    groupBox1_2Layout->setAlignment( Qt::AlignTop );

    itsTestingTargetObject = new QLineEdit( groupBox1_2, "itsTestingTargetObject" );

    groupBox1_2Layout->addMultiCellWidget( itsTestingTargetObject, 1, 1, 1, 6 );

    itsTestingSceneType = new QComboBox( FALSE, groupBox1_2, "itsTestingSceneType" );

    groupBox1_2Layout->addMultiCellWidget( itsTestingSceneType, 0, 0, 1, 6 );

    textLabel1_2 = new QLabel( groupBox1_2, "textLabel1_2" );

    groupBox1_2Layout->addWidget( textLabel1_2, 0, 0 );

    textLabel2_2 = new QLabel( groupBox1_2, "textLabel2_2" );

    groupBox1_2Layout->addWidget( textLabel2_2, 1, 0 );

    textLabel7_2 = new QLabel( groupBox1_2, "textLabel7_2" );

    groupBox1_2Layout->addWidget( textLabel7_2, 2, 0 );

    textLabel4_2 = new QLabel( groupBox1_2, "textLabel4_2" );

    groupBox1_2Layout->addWidget( textLabel4_2, 3, 0 );

    itsTestingRotation = new QLineEdit( groupBox1_2, "itsTestingRotation" );
    itsTestingRotation->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingRotation->sizePolicy().hasHeightForWidth() ) );
    itsTestingRotation->setMaximumSize( QSize( 30, 23 ) );
    itsTestingRotation->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingRotation, 6, 2 );

    textLabel5_2_2 = new QLabel( groupBox1_2, "textLabel5_2_2" );

    groupBox1_2Layout->addWidget( textLabel5_2_2, 6, 0 );

    textLabel5_3 = new QLabel( groupBox1_2, "textLabel5_3" );

    groupBox1_2Layout->addWidget( textLabel5_3, 7, 0 );

    itsTestingNoise = new QLineEdit( groupBox1_2, "itsTestingNoise" );
    itsTestingNoise->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingNoise->sizePolicy().hasHeightForWidth() ) );
    itsTestingNoise->setMaximumSize( QSize( 30, 23 ) );
    itsTestingNoise->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingNoise, 7, 2 );

    textLabel6_2 = new QLabel( groupBox1_2, "textLabel6_2" );

    groupBox1_2Layout->addWidget( textLabel6_2, 4, 0 );

    textLabel3_2 = new QLabel( groupBox1_2, "textLabel3_2" );

    groupBox1_2Layout->addWidget( textLabel3_2, 5, 0 );

    textLabel10_3 = new QLabel( groupBox1_2, "textLabel10_3" );

    groupBox1_2Layout->addWidget( textLabel10_3, 2, 5 );

    itsTestingColorB = new QLineEdit( groupBox1_2, "itsTestingColorB" );
    itsTestingColorB->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingColorB->sizePolicy().hasHeightForWidth() ) );
    itsTestingColorB->setMaximumSize( QSize( 30, 23 ) );
    itsTestingColorB->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingColorB, 3, 6 );

    textLabel10_2_3 = new QLabel( groupBox1_2, "textLabel10_2_3" );

    groupBox1_2Layout->addWidget( textLabel10_2_3, 3, 5 );

    textLabel8_2_3 = new QLabel( groupBox1_2, "textLabel8_2_3" );

    groupBox1_2Layout->addWidget( textLabel8_2_3, 3, 1 );

    textLabel9_2_2_2 = new QLabel( groupBox1_2, "textLabel9_2_2_2" );

    groupBox1_2Layout->addWidget( textLabel9_2_2_2, 4, 3 );

    textLabel9_2_3 = new QLabel( groupBox1_2, "textLabel9_2_3" );

    groupBox1_2Layout->addWidget( textLabel9_2_3, 3, 3 );

    itsTestingBackColorG = new QLineEdit( groupBox1_2, "itsTestingBackColorG" );
    itsTestingBackColorG->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingBackColorG->sizePolicy().hasHeightForWidth() ) );
    itsTestingBackColorG->setMaximumSize( QSize( 30, 23 ) );
    itsTestingBackColorG->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingBackColorG, 4, 4 );

    textLabel9_3 = new QLabel( groupBox1_2, "textLabel9_3" );

    groupBox1_2Layout->addWidget( textLabel9_3, 2, 3 );

    itsTestingTargetColorG = new QLineEdit( groupBox1_2, "itsTestingTargetColorG" );
    itsTestingTargetColorG->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingTargetColorG->sizePolicy().hasHeightForWidth() ) );
    itsTestingTargetColorG->setMaximumSize( QSize( 30, 23 ) );
    itsTestingTargetColorG->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingTargetColorG, 2, 4 );

    itsTestingColorG = new QLineEdit( groupBox1_2, "itsTestingColorG" );
    itsTestingColorG->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingColorG->sizePolicy().hasHeightForWidth() ) );
    itsTestingColorG->setMaximumSize( QSize( 30, 23 ) );
    itsTestingColorG->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingColorG, 3, 4 );

    textLabel10_2_2_2 = new QLabel( groupBox1_2, "textLabel10_2_2_2" );

    groupBox1_2Layout->addWidget( textLabel10_2_2_2, 4, 5 );

    itsTestingTargetColorB = new QLineEdit( groupBox1_2, "itsTestingTargetColorB" );
    itsTestingTargetColorB->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingTargetColorB->sizePolicy().hasHeightForWidth() ) );
    itsTestingTargetColorB->setMaximumSize( QSize( 30, 23 ) );
    itsTestingTargetColorB->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingTargetColorB, 2, 6 );

    itsTestingBackColorR = new QLineEdit( groupBox1_2, "itsTestingBackColorR" );
    itsTestingBackColorR->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingBackColorR->sizePolicy().hasHeightForWidth() ) );
    itsTestingBackColorR->setMaximumSize( QSize( 30, 23 ) );
    itsTestingBackColorR->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingBackColorR, 4, 2 );

    textLabel8_2_2_2 = new QLabel( groupBox1_2, "textLabel8_2_2_2" );

    groupBox1_2Layout->addWidget( textLabel8_2_2_2, 4, 1 );

    itsTestingBackColorB = new QLineEdit( groupBox1_2, "itsTestingBackColorB" );
    itsTestingBackColorB->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingBackColorB->sizePolicy().hasHeightForWidth() ) );
    itsTestingBackColorB->setMaximumSize( QSize( 30, 23 ) );
    itsTestingBackColorB->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingBackColorB, 4, 6 );

    itsTestingColorR = new QLineEdit( groupBox1_2, "itsTestingColorR" );
    itsTestingColorR->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingColorR->sizePolicy().hasHeightForWidth() ) );
    itsTestingColorR->setMaximumSize( QSize( 30, 23 ) );
    itsTestingColorR->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingColorR, 3, 2 );

    textLabel8_3 = new QLabel( groupBox1_2, "textLabel8_3" );

    groupBox1_2Layout->addWidget( textLabel8_3, 2, 1 );

    itsTestingLum = new QLineEdit( groupBox1_2, "itsTestingLum" );
    itsTestingLum->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingLum->sizePolicy().hasHeightForWidth() ) );
    itsTestingLum->setMaximumSize( QSize( 30, 23 ) );
    itsTestingLum->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingLum, 5, 2 );

    itsTestingTargetColorR = new QLineEdit( groupBox1_2, "itsTestingTargetColorR" );
    itsTestingTargetColorR->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, itsTestingTargetColorR->sizePolicy().hasHeightForWidth() ) );
    itsTestingTargetColorR->setMaximumSize( QSize( 30, 23 ) );
    itsTestingTargetColorR->setMaxLength( 3 );

    groupBox1_2Layout->addWidget( itsTestingTargetColorR, 2, 2 );

    SceneSettingsDialogLayout->addWidget( groupBox1_2, 0, 1 );

    Layout1 = new QHBoxLayout( 0, 0, 6, "Layout1");
    Horizontal_Spacing2 = new QSpacerItem( 20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    Layout1->addItem( Horizontal_Spacing2 );

    buttonOk = new QPushButton( this, "buttonOk" );
    buttonOk->setAutoDefault( TRUE );
    buttonOk->setDefault( TRUE );
    Layout1->addWidget( buttonOk );

    buttonCancel = new QPushButton( this, "buttonCancel" );
    buttonCancel->setAutoDefault( TRUE );
    Layout1->addWidget( buttonCancel );

    SceneSettingsDialogLayout->addMultiCellLayout( Layout1, 1, 1, 0, 1 );
    languageChange();
    resize( QSize(550, 324).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( buttonOk, SIGNAL( clicked() ), this, SLOT( accept() ) );
    connect( buttonCancel, SIGNAL( clicked() ), this, SLOT( reject() ) );

    // tab order
    setTabOrder( itsTrainingSceneType, itsTrainingTargetObject );
    setTabOrder( itsTrainingTargetObject, itsTrainingTargetColorR );
    setTabOrder( itsTrainingTargetColorR, itsTrainingTargetColorG );
    setTabOrder( itsTrainingTargetColorG, itsTrainingTargetColorB );
    setTabOrder( itsTrainingTargetColorB, itsTrainingLum );
    setTabOrder( itsTrainingLum, itsTrainingColorR );
    setTabOrder( itsTrainingColorR, itsTrainingColorG );
    setTabOrder( itsTrainingColorG, itsTrainingColorB );
    setTabOrder( itsTrainingColorB, itsTrainingRotation );
    setTabOrder( itsTrainingRotation, itsTrainingNoise );
    setTabOrder( itsTrainingNoise, itsTrainingBackColorR );
    setTabOrder( itsTrainingBackColorR, itsTrainingBackColorG );
    setTabOrder( itsTrainingBackColorG, itsTrainingBackColorB );
    setTabOrder( itsTrainingBackColorB, buttonOk );
    setTabOrder( buttonOk, buttonCancel );
}

/*
 *  Destroys the object and frees any allocated resources
 */
SceneSettingsDialog::~SceneSettingsDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void SceneSettingsDialog::languageChange()
{
    setCaption( tr( "Scene Settings" ) );
    groupBox1->setTitle( tr( "Training Scene Settings" ) );
    textLabel1->setText( tr( "Scene Type:" ) );
    textLabel2->setText( tr( "Target Object:" ) );
    textLabel7->setText( tr( "Target Color:" ) );
    textLabel4->setText( tr( "Color:" ) );
    itsTrainingRotation->setInputMask( tr( "###; " ) );
    textLabel5_2->setText( tr( "Rotation:" ) );
    textLabel5->setText( tr( "Pixel Noise:" ) );
    itsTrainingNoise->setInputMask( tr( "###; " ) );
    textLabel6->setText( tr( "Background Color:" ) );
    textLabel3->setText( tr( "Illumination:" ) );
    textLabel10->setText( tr( "B:" ) );
    itsTrainingColorB->setInputMask( tr( "###; " ) );
    textLabel10_2->setText( tr( "B:" ) );
    textLabel8_2->setText( tr( "R:" ) );
    textLabel9_2_2->setText( tr( "G:" ) );
    textLabel9_2->setText( tr( "G:" ) );
    itsTrainingBackColorG->setInputMask( tr( "###; " ) );
    textLabel9->setText( tr( "G:" ) );
    itsTrainingTargetColorG->setInputMask( tr( "###; " ) );
    itsTrainingColorG->setInputMask( tr( "###; " ) );
    textLabel10_2_2->setText( tr( "B:" ) );
    itsTrainingTargetColorB->setInputMask( tr( "###; " ) );
    itsTrainingBackColorR->setInputMask( tr( "###; " ) );
    textLabel8_2_2->setText( tr( "R:" ) );
    itsTrainingBackColorB->setInputMask( tr( "###; " ) );
    itsTrainingColorR->setInputMask( tr( "###; " ) );
    textLabel8->setText( tr( "R:" ) );
    itsTrainingLum->setInputMask( tr( "###; " ) );
    itsTrainingTargetColorR->setInputMask( tr( "###; " ) );
    groupBox1_2->setTitle( tr( "Testing Scene Settings" ) );
    textLabel1_2->setText( tr( "Scene Type:" ) );
    textLabel2_2->setText( tr( "Target Object:" ) );
    textLabel7_2->setText( tr( "Target Color:" ) );
    textLabel4_2->setText( tr( "Color:" ) );
    itsTestingRotation->setInputMask( tr( "###; " ) );
    textLabel5_2_2->setText( tr( "Rotation:" ) );
    textLabel5_3->setText( tr( "Pixel Noise:" ) );
    itsTestingNoise->setInputMask( tr( "###; " ) );
    textLabel6_2->setText( tr( "Background Color:" ) );
    textLabel3_2->setText( tr( "Illumination:" ) );
    textLabel10_3->setText( tr( "B:" ) );
    itsTestingColorB->setInputMask( tr( "###; " ) );
    textLabel10_2_3->setText( tr( "B:" ) );
    textLabel8_2_3->setText( tr( "R:" ) );
    textLabel9_2_2_2->setText( tr( "G:" ) );
    textLabel9_2_3->setText( tr( "G:" ) );
    itsTestingBackColorG->setInputMask( tr( "###; " ) );
    textLabel9_3->setText( tr( "G:" ) );
    itsTestingTargetColorG->setInputMask( tr( "###; " ) );
    itsTestingColorG->setInputMask( tr( "###; " ) );
    textLabel10_2_2_2->setText( tr( "B:" ) );
    itsTestingTargetColorB->setInputMask( tr( "###; " ) );
    itsTestingBackColorR->setInputMask( tr( "###; " ) );
    textLabel8_2_2_2->setText( tr( "R:" ) );
    itsTestingBackColorB->setInputMask( tr( "###; " ) );
    itsTestingColorR->setInputMask( tr( "###; " ) );
    textLabel8_3->setText( tr( "R:" ) );
    itsTestingLum->setInputMask( tr( "###; " ) );
    itsTestingTargetColorR->setInputMask( tr( "###; " ) );
    buttonOk->setText( tr( "&OK" ) );
    buttonOk->setAccel( QKeySequence( QString::null ) );
    buttonCancel->setText( tr( "&Cancel" ) );
    buttonCancel->setAccel( QKeySequence( QString::null ) );
}

