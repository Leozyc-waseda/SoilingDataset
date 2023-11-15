

 class State{
         public :
                 void State(Martix& m) ;
                 void ~State();
                 std::vector<Action*>* getActionsList() ;
                 Matrix* getMatrix() ;
         private :
                 Martix* matrix ;
                 std::vector<Action*>* actionsList ;
 };

 class Action{
         public :
                 void Action(Cell& cell1, Cell& cell2);
                 void ~Action ;
                 std::map<State,double>* getStatesMap();
                  Cell* c1,c2;
         private :
                 std::map<State*,double>* statesMap ;
 };

 std::vector<Action>* findTheActions(Matrix& m);