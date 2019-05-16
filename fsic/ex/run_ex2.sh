#!/bin/bash 

screen -AdmS ex2_fsic -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex2_fsic -X screen -t tab1 bash -lic "python ex2_prob_params.py sin"
screen -S ex2_fsic -X screen -t tab2 bash -lic "python ex2_prob_params.py gsign"
screen -S ex2_fsic -X screen -t tab3 bash -lic "python ex2_prob_params.py bsg"

#screen -S ex2_fsic -X screen -t tab2 bash -lic "python ex2_prob_params.py sg"
#screen -S ex2_fsic -X screen -t tab4 bash -lic "python ex2_prob_params.py msin"
#screen -S ex2_fsic -X screen -t tab3 bash -lic "python ex2_prob_params.py pwsign"
#screen -S ex2_fsic -X screen -t tab3 bash -lic "python ex2_prob_params.py u2drot"
#screen -S ex2_fsic -X screen -t tab4 bash -lic "python ex2_prob_params.py urot_noise"

#python ex2_prob_params.py urot_noise
#python ex2_prob_params.py sg
#python ex2_prob_params.py u2drot
#python ex2_prob_params.py sin
