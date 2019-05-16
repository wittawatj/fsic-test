#!/bin/bash 

screen -AdmS ex4_fsic -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

#screen -S ex4_fsic -X screen -t tab1 bash -lic "python ex4_real_data.py wine_quality white_wine.n2000"
#screen -S ex4_fsic -X screen -t tab2 bash -lic "python ex4_real_data.py wine_quality white_wine_h0.n2000"
screen -S ex4_fsic -X screen -t tab1 bash -lic "python ex4_real_data.py wine_quality white_wine_std.n2000"
screen -S ex4_fsic -X screen -t tab2 bash -lic "python ex4_real_data.py wine_quality white_wine_std_h0.n2000"
screen -S ex4_fsic -X screen -t tab3 bash -lic "python ex4_real_data.py msd msd50000_std.n2000"
screen -S ex4_fsic -X screen -t tab3 bash -lic "python ex4_real_data.py msd msd50000_std_h0.n2000"
#screen -S ex4_fsic -X screen -t tab4 bash -lic "python ex4_real_data.py wine_quality white_wine_std_h0.n3000"

#screen -S ex4_fsic -X screen -t tab2 bash -lic "python ex2_prob_params.py gsign"
#screen -S ex4_fsic -X screen -t tab3 bash -lic "python ex2_prob_params.py bsg"

