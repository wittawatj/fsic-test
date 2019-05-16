#!/bin/bash 

screen -AdmS ex5_fsic -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py msd msd50000_std"
screen -S ex5_fsic -X screen -t tab4 bash -lic "python ex5_real_vary_n.py msd msd50000_std_h0"

screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py videostory46k data_n10000_td1878_vd2000_std"
screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py videostory46k data_n10000_td1878_vd2000_std_h0"

#python ex5_real_vary_n.py msd msd50000_std
#python ex5_real_vary_n.py msd msd50000_std_h0
#python ex5_real_vary_n.py videostory46k data_n10000_td1878_vd2000_std
#python ex5_real_vary_n.py videostory46k data_n10000_td1878_vd2000_std_h0



# -------------------------------

#screen -S ex5_fsic -X screen -t tab1 bash -lic "python ex5_real_vary_n.py wine_quality white_wine_ndx5_ndy5_std"
#screen -S ex5_fsic -X screen -t tab2 bash -lic "python ex5_real_vary_n.py wine_quality white_wine_ndx5_ndy5_std_h0"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py news_popularity news_popularity_ndx5_ndy5_std"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py news_popularity news_popularity_ndx5_ndy5_std_h0"

#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py movie_rating movie_rating_std"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py arizona_fs lung_c_std"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py arizona_fs carcinom_c_std"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py arizona_fs CLL_SUB_111_c_std"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py arizona_fs SMK_CAN_187_c_std"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py arizona_fs TOX_171_c_std"

#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py higgs higgs_no_deriv_c_std"


#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py earth_temperature latlong_temp_y2013_std"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py voice_gender voice_gender_c_std"


#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py geographical_music music68_std"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py geographical_music music68_std_h0"


#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py geographical_music chromatic_music_std"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py geographical_music chromatic_music_std_h0"


#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py skillcraft1 skillcraft1_ndx10_ndy10_std"
#screen -S ex5_fsic -X screen -t tab3 bash -lic "python ex5_real_vary_n.py skillcraft1 skillcraft1_ndx10_ndy10_std_h0"



