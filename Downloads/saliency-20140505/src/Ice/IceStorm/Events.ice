
module EventsNS {
  class Event {
    int id;
  };

  class Message1 extends Event {
    int m;
    string msg;
  };

  class Message2  extends Event {
    int i;
    int j;
    string msg;
  };

  interface Events {
    ["ami"] void evolve (Event e);
  };

};

module TestPub {
  interface Publisher1 extends EventsNS::Events {
  };

  interface Publisher2 extends EventsNS::Events {
  };

  interface Subscriber1 extends EventsNS::Events {
  };

  interface Subscriber2 extends EventsNS::Events {
  };
};



