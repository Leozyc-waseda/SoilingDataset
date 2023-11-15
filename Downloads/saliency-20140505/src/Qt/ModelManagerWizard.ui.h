/*! @file Qt/ModelManagerWizard.ui.h functions relating to model configuration wizard */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/ModelManagerWizard.ui.h $
// $Id: ModelManagerWizard.ui.h 7381 2006-11-01 18:56:25Z rjpeters $

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

namespace dummy_namespace_to_avoid_gcc411_bug_ModelManagerWizard_ui_h
{
  typedef std::map<const ModelOptionCateg*, ModelManagerWizardItem> categs_type;

  struct DefInfo
  {
    int ypos;
  };
}

using namespace dummy_namespace_to_avoid_gcc411_bug_ModelManagerWizard_ui_h;

class ModelManagerWizardItem
{
public:
  ModelManagerWizardItem() :
    lbitem(0), sv(0), count(0)
  {}

  QListBoxItem* lbitem;
  QScrollView* sv;
  std::map<const ModelOptionDef*, DefInfo> defs;
  int count;
};

void ModelManagerWizard::init(ModelManager & manager)
{
  errBox = new QErrorMessage(this);
  mgr = &manager;
  mgr->writeParamsTo(backupMap);
  refreshOptions();
}

void ModelManagerWizard::refreshOptions(void)
{
  currentSv = 0;
  categs.clear();
  listbox->clear();
  listbox->triggerUpdate(false);

  const uint ndefs = mgr->numOptionDefs();

  const ModelOptionDef* defs[ndefs];

  const uint nfilled = mgr->getOptionDefs(defs, ndefs);

  // determine which option categories are requested
  for (uint i = 0; i < nfilled; ++i)
    {
      const ModelOptionDef* def = defs[i];

      if (!mgr->isOptionDefUsed(def))
        continue;

      if (categs.find(def->categ) == categs.end())
        {
          ModelManagerWizardItem newitem;

          newitem.lbitem =
            new QListBoxText(listbox, def->categ->description);

          newitem.sv = new QScrollView(this);
          newitem.sv->setHScrollBarMode(QScrollView::AlwaysOff);
          newitem.sv->setGeometry(260, 10, 500, 420);
          newitem.sv->viewport()->setPaletteBackgroundColor(this->paletteBackgroundColor());
          newitem.sv->enableClipper(true);
          newitem.sv->hide();

          categs.insert(categs_type::value_type(def->categ, newitem));
        }

      ModelManagerWizardItem& item = categs[def->categ];
      ASSERT(item.lbitem != 0);
      ASSERT(item.sv != 0);

      item.defs[def].ypos = 20 + 40 * item.count;

      // add option label
      QLabel* label = new QLabel(def->longoptname, item.sv->viewport());
      label->adjustSize();
      QWhatsThis::add(label, def->descr);
      item.sv->addChild(label, 20, item.defs[def].ypos);

      // add option input widget based on what ModelOptionType it is
      const std::type_info* argtype = def->type.argtype;

      if (argtype != 0 && *argtype == typeid(bool))
        {
          QCheckBox* cb = new QCheckBox(item.sv->viewport(), def->name);
          cb->setChecked(fromStr<bool>(mgr->getOptionValString(def)));
          connect(cb, SIGNAL(toggled(bool)),
                  this, SLOT(handleCheckBox(bool)));
          itsWidgetOptions[cb] = def;
          item.sv->addChild(cb, 300, item.defs[def].ypos);
        }
      else
        {
          // for now, treat everything else as a string
          QLineEdit* le = new QLineEdit(item.sv->viewport(), def->name);
          le->setText(mgr->getOptionValString(def));
          connect(le, SIGNAL(lostFocus(void)),
                  this, SLOT(handleLineEdit(void)));
          itsWidgetOptions[le] = def;
          item.sv->addChild(le, 300, item.defs[def].ypos);
        }

      ++item.count;
      item.sv->resizeContents(400, 20 + item.defs[def].ypos);
    }

  QListBoxItem* first = listbox->item(0);
  listbox->setCurrentItem(first);
  showFrame(first);
}

void ModelManagerWizard::showFrame(QListBoxItem* item)
{
  for (categs_type::iterator itr = categs.begin();
       itr != categs.end(); ++itr)
    if ((*itr).second.lbitem == item)
      {
        // switch frame
        if (currentSv != 0) currentSv->hide();
        currentSv = (*itr).second.sv;
        currentSv->show();

        const int index = listbox->index((*itr).second.lbitem);

        // hide back button if this is first frame
        if   (index == 0) backButton->hide();
        else              backButton->show();

        // hide next button if this is last frame
        if   (index == listbox->numRows() - 1) nextButton->hide();
        else                                   nextButton->show();
      }
}

// handle buttons on Wizard form

void ModelManagerWizard::handleCancelButton(void)
{
  mgr->readParamsFrom(backupMap);
  close();
}

void ModelManagerWizard::handleBackButton(void)
{
  listbox->setCurrentItem(listbox->currentItem() - 1);
  showFrame(listbox->selectedItem());
}

void ModelManagerWizard::handleNextButton(void)
{
  listbox->setCurrentItem(listbox->currentItem() + 1);
  showFrame(listbox->selectedItem());
}

void ModelManagerWizard::handleFinishButton(void)
{
  close();
}

// handle signals from input widgets
void ModelManagerWizard::handleCheckBox(bool b)
{
  const QCheckBox* box = dynamic_cast<const QCheckBox*>(sender());
  ASSERT(box != 0);

  const ModelOptionDef* def = itsWidgetOptions[box];
  ASSERT(def != 0);

  try
    {
      mgr->setOptionValString(def, toStr(b));
    }
  catch (std::exception& e)
    {
      errBox->message(e.what());
    }
  refreshAndSelect(sender()->name(), def);
}

void ModelManagerWizard::handleLineEdit(void)
{
  const QLineEdit* line = dynamic_cast<const QLineEdit*>(sender());
  ASSERT(line != 0);

  const ModelOptionDef* def = itsWidgetOptions[line];
  ASSERT(def != 0);

  try
    {
      mgr->setOptionValString(def, line->text().latin1());
    }
  catch (std::exception& e)
    {
      errBox->message(e.what());
    }
  refreshAndSelect(sender()->name(), def);
}

void ModelManagerWizard::refreshAndSelect(QString sel,
                                          const ModelOptionDef* def)
{
  // reload options in case new ones added/old ones removed
  refreshOptions();

  // locate field that was changed and make it visible
  categs_type::iterator itr = categs.find(def->categ);
  ASSERT(itr != categs.end());

  // show the frame that field belongs to
  listbox->setCurrentItem((*itr).second.lbitem);
  showFrame((*itr).second.lbitem);

  // locate field in frame and ensure visible
  (*itr).second.sv->ensureVisible(300, (*itr).second.defs[def].ypos);
}
