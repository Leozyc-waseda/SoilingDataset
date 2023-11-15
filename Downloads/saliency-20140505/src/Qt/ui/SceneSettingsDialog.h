/****************************************************************************
** Form interface generated from reading ui file 'Qt/SceneSettingsDialog.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef SCENESETTINGSDIALOG_H
#define SCENESETTINGSDIALOG_H

#include <qvariant.h>
#include <qdialog.h>
#include "Media/SceneGenerator.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QGroupBox;
class QLineEdit;
class QComboBox;
class QLabel;
class QPushButton;

class SceneSettingsDialog : public QDialog
{
    Q_OBJECT

public:
    SceneSettingsDialog( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~SceneSettingsDialog();

    QGroupBox* groupBox1;
    QLineEdit* itsTrainingTargetObject;
    QComboBox* itsTrainingSceneType;
    QLabel* textLabel1;
    QLabel* textLabel2;
    QLabel* textLabel7;
    QLabel* textLabel4;
    QLineEdit* itsTrainingRotation;
    QLabel* textLabel5_2;
    QLabel* textLabel5;
    QLineEdit* itsTrainingNoise;
    QLabel* textLabel6;
    QLabel* textLabel3;
    QLabel* textLabel10;
    QLineEdit* itsTrainingColorB;
    QLabel* textLabel10_2;
    QLabel* textLabel8_2;
    QLabel* textLabel9_2_2;
    QLabel* textLabel9_2;
    QLineEdit* itsTrainingBackColorG;
    QLabel* textLabel9;
    QLineEdit* itsTrainingTargetColorG;
    QLineEdit* itsTrainingColorG;
    QLabel* textLabel10_2_2;
    QLineEdit* itsTrainingTargetColorB;
    QLineEdit* itsTrainingBackColorR;
    QLabel* textLabel8_2_2;
    QLineEdit* itsTrainingBackColorB;
    QLineEdit* itsTrainingColorR;
    QLabel* textLabel8;
    QLineEdit* itsTrainingLum;
    QLineEdit* itsTrainingTargetColorR;
    QGroupBox* groupBox1_2;
    QLineEdit* itsTestingTargetObject;
    QComboBox* itsTestingSceneType;
    QLabel* textLabel1_2;
    QLabel* textLabel2_2;
    QLabel* textLabel7_2;
    QLabel* textLabel4_2;
    QLineEdit* itsTestingRotation;
    QLabel* textLabel5_2_2;
    QLabel* textLabel5_3;
    QLineEdit* itsTestingNoise;
    QLabel* textLabel6_2;
    QLabel* textLabel3_2;
    QLabel* textLabel10_3;
    QLineEdit* itsTestingColorB;
    QLabel* textLabel10_2_3;
    QLabel* textLabel8_2_3;
    QLabel* textLabel9_2_2_2;
    QLabel* textLabel9_2_3;
    QLineEdit* itsTestingBackColorG;
    QLabel* textLabel9_3;
    QLineEdit* itsTestingTargetColorG;
    QLineEdit* itsTestingColorG;
    QLabel* textLabel10_2_2_2;
    QLineEdit* itsTestingTargetColorB;
    QLineEdit* itsTestingBackColorR;
    QLabel* textLabel8_2_2_2;
    QLineEdit* itsTestingBackColorB;
    QLineEdit* itsTestingColorR;
    QLabel* textLabel8_3;
    QLineEdit* itsTestingLum;
    QLineEdit* itsTestingTargetColorR;
    QPushButton* buttonOk;
    QPushButton* buttonCancel;

    virtual void init( SceneGenerator * trainScene, SceneGenerator * testScene );

public slots:
    virtual void accept();

protected:
    SceneGenerator *itsTestScene;
    SceneGenerator *itsTrainScene;

    QGridLayout* SceneSettingsDialogLayout;
    QGridLayout* groupBox1Layout;
    QGridLayout* groupBox1_2Layout;
    QHBoxLayout* Layout1;
    QSpacerItem* Horizontal_Spacing2;

protected slots:
    virtual void languageChange();

};

#endif // SCENESETTINGSDIALOG_H
