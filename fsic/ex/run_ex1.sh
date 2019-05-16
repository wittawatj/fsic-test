#!/bin/bash 

screen -AdmS ex1_fsic -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex1_fsic -X screen -t tab1 bash -lic "python ex1_vary_n.py sg_d250"
screen -S ex1_fsic -X screen -t tab2 bash -lic "python ex1_vary_n.py sin_w4"
screen -S ex1_fsic -X screen -t tab3 bash -lic "python ex1_vary_n.py gsign_d4"
#screen -S ex1_fsic -X screen -t tab4 bash -lic "python ex1_vary_n.py sg_d50"
#screen -S ex1_fsic -X screen -t tab2 bash -lic "python ex2_prob_params.py gsign"
#python ex1_vary_n.py sg_d50
