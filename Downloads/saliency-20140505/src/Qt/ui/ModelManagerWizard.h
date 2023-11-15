/****************************************************************************
** Form interface generated from reading ui file 'Qt/ModelManagerWizard.ui'
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef MODELMANAGERWIZARD_H
#define MODELMANAGERWIZARD_H

#include <qvariant.h>
#include <qdialog.h>
#include <qscrollview.h>
#include <map>
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Component/ParamMap.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QPushButton;
class QLabel;
class QListBox;
class QListBoxItem;
class QErrorMessage;
class ModelManagerWizardItem;

class ModelManagerWizard : public QDialog
{
    Q_OBJECT

public:
    ModelManagerWizard( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~ModelManagerWizard();

    QPushButton* cancelButton;
    QPushButton* nextButton;
    QPushButton* backButton;
    QPushButton* finishButton;
    QLabel* textLabel2;
    QListBox* listbox;

public slots:
    virtual void init( ModelManager & manager );
    virtual void refreshOptions( void );
    virtual void showFrame( QListBoxItem * item );
    virtual void handleCancelButton( void );
    virtual void handleBackButton( void );
    virtual void handleNextButton( void );
    virtual void handleFinishButton( void );
    virtual void handleCheckBox( bool b );
    virtual void handleLineEdit( void );
    virtual void refreshAndSelect(QString sel, const ModelOptionDef* def);

protected:

protected slots:
    virtual void languageChange();

private:
    QErrorMessage* errBox;
    QScrollView* currentSv;
    std::map<const ModelOptionCateg*, ModelManagerWizardItem> categs;
    std::map<const QWidget*, const ModelOptionDef*> itsWidgetOptions;
    ModelManager* mgr;
    ParamMap backupMap;

};

#endif // MODELMANAGERWIZARD_H
