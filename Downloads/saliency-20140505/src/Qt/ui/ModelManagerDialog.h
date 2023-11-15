/****************************************************************************
** Form interface generated from reading ui file 'Qt/ModelManagerDialog.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef MODELMANAGERDIALOG_H
#define MODELMANAGERDIALOG_H

#include <qvariant.h>
#include <qdialog.h>
#include "Component/ModelManager.H"
#include "Component/ParamMap.H"
#include "ModelManagerWizard.h"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QLabel;
class QPushButton;
class QListView;
class QListViewItem;

class ModelManagerDialog : public QDialog
{
    Q_OBJECT

public:
    ModelManagerDialog( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~ModelManagerDialog();

    QLabel* textLabel1;
    QPushButton* wizardButton;
    QPushButton* cancelButton;
    QPushButton* applyButton;
    QListView* listview;

    virtual void init( ModelManager & manager );
    virtual void populate( rutz::shared_ptr<ParamMap> pmp, QListViewItem * parent );

public slots:
    virtual void handleItemEdit( QListViewItem * item );
    virtual void handleWizardButton( void );
    virtual void handleApplyButton( void );
    virtual void handleCancelButton( void );

protected:

protected slots:
    virtual void languageChange();

private:
    ParamMap backupMap;
    ParamMap pmap;
    ModelManager *mgr;
    ModelManagerWizard mmw;

};

#endif // MODELMANAGERDIALOG_H
