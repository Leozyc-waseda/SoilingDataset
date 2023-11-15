
//#######################################################################
void State::State(Matrix& m){
        matrix = &m;
}
//#######################################################################
void State::~State(){
        for( int i = 0 ; i < actionsList->size() ; i++){
                delete (*actionsList)[i] ;
        }
        delete actionsList;
}
//#######################################################################
std::vector<Action*>* State::getActionsList(){
        if (actions == 0 ) actionsList = findTheActions(*matrix);
        return actionsList ;
}
//#######################################################################
Matrix* State::getMatrix(){
        return matrix ;
}
//#######################################################################

void Action::Action( Cell& cell1, Cell& cell2){
        c1 = &cell1 ;
        c2 = &cell2 ;
}

//#######################################################################

void Action::~Action(){
        if(c1!=0){
                delete c1 ;
                delete c2 ;
        }
}

//#######################################################################
std::vector<Action*>* findTheActions(Matrix& m){

}
//#######################################################################



