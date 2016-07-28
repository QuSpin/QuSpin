for L in $(seq 10 1 16) #loop over system sizes
do


    for Omega in $(seq 5 1 15) #loop over frequencies
    do

        for k in $(seq 1 1 $((1+(L+2)/2)) ) #loop over symm blocks
        do


        echo "#!/bin/bash -login" > submission.sh
        echo "#PBS -N evolve_${L}_${Omega}_${k}" >> submission.sh #evolve is name of job
        echo "#PBS -j oe" >> submission.sh #ouput and error files together
        echo "#PBS -l mem=1000mb" >> submission.sh # L=16: 2500
        # echo "#PBS -N evolve  >> submission.sh
        echo "#PBS -m ae" >> submission.sh #a (abort) e(exit)
        #echo "#PBS -M mbukov@bu.edu" >> submission.sh
        #echo "source activate exact_diag" >> submission.sh
        echo "cd \$PBS_O_WORKDIR" >> submission.sh #cd filepath to correct dir
        echo "python spin_only.py $L $Omega $k" >> submission.sh
        qsub submission.sh
        rm submission.sh

        sleep 2


        done

  
    done


done