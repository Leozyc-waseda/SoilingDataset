# run me with bin/invt

# $Id: ezvision.tcl 11019 2009-03-11 20:57:11Z itti $

proc main {} {

    set mm [ModelManager::find_all]

    #-> $mm param DebugMode 1

    set seq [new SimEventQueue]
    -> $mm addSubComponent $seq true

    # NOTE: make sure you register your OutputFrameSeries with the
    # manager before you do your InputFrameSeries, to ensure that
    # outputs for the current frame get saved before the next input
    # frame is loaded.
    set ofs [new SimOutputFrameSeries]
    -> $mm addSubComponent $ofs true

    set ifs [new SimInputFrameSeries]
    -> $mm addSubComponent $ifs true

    set brain [new StdBrain]
    -> $mm addSubComponent $brain true

    -> $mm parseCommandLine [concat $::argv0 $::argv] "" 0 0

    -> $mm start

    while { 1 } {

        set should_break [SimEventQueue::evolveMaster $seq]

        if { $should_break != 0 } {
            break
        }
    }
}

main
