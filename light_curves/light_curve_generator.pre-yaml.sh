#!/bin/bash

# [TODO] append to all log and data files, don't overwrite

# ---- Define some functions.
kill_all_pids(){
    run_id=$1
    logsdir=$2
    logfile=$3

    # scrape log files and collect all PIDs into an array
    for file in "$logsdir"/*.log; do
        # use regex to match the number in a string with the syntax [pid=1234]
        # https://unix.stackexchange.com/questions/13466/can-grep-output-only-specified-groupings-that-match
        # ----
        # [TODO] THIS WILL FAIL ON MacOS (invalid option '-P') but I haven't found a more general solution
        # https://stackoverflow.com/questions/77662026/grep-invalid-option-p-error-when-doing-regex-in-bash-script
        # (the '-P' is required for the look-behind, '\K', which excludes "[pid=" from the returned result)
        # ----
        all_pids+=($(grep -oP '\[pid=\K\d+' $file))
    done
    
    # deduplicate the array
    pids=($(for pid in "${all_pids[@]}"; do echo $pid; done | sort --sort=numeric --unique))
    # add currently running python PIDs
    # https://stackoverflow.com/questions/21470362/find-the-pids-of-running-processes-and-store-as-an-array
    pids+=($(ps -ef | grep python | awk '{print $2}'))
    # get only values that are in both lists
    kill_pids=($(for pid in "${pids[@]}"; do echo $pid; done | sort | uniq --repeated))

    # killing processes can be dangerous. make the user confirm.
    echo "WARNING, you are about to kill all processes started by the run run_id='${run_id}'." | tee -a ${logfile}
    echo "This includes the following PIDs: ${kill_pids[@]}" | tee -a ${logfile}
    read -p "Enter 'y' to continue or any other key to abort: " continue_kill
    continue_kill="${continue_kill:-n}"
    if [ $continue_kill == "y" ]; then
        echo "Killing." | tee -a ${logfile}
        # kill each PID
        for pid in "${kill_pids[@]}"; do kill $pid; done
    else
        echo "Aborting." | tee -a ${logfile}
    fi

}

print_help_me(){
    echo "For help, call this script with the '-h' flag:"
    echo "    \$ ./$(basename $0) -h"
}

print_help_run_instructions(){
    echo "For instructions on monitoring a run and loading the output, use:"
    echo "    \$ ./$(basename $0) -i"
}

print_logs(){
    logfile=$1
    echo "-- ${logfile}:"
    cat $logfile
    echo "--"
}

print_run_instructions(){
    echo
    echo "---- Check Progress ----"
    echo "When this script is executed it launches a run consisting of multiple jobs:"
    echo "    - one for this script"
    echo "    - one for the function that builds the object sample"
    echo "    - one for each call to a mission archive"
    echo "There are several ways to check the progress, listed below."
    echo
    echo "1: Logs"
    echo "To check a job's progress, view its log file by setting the 'logfile' variable (you can"
    echo "copy/paste this from the script's output), then using:"
    echo "    \$ cat \$logfile"
    echo "View this script's log for high-level job info and variables to copy/paste."
    echo "View a mission's log to check its progress."
    echo
    echo "2: 'top'"
    echo "Use 'top' to monitor job activity (job PID's are in script output) and overall resource usage."
    echo
    echo "3: Output"
    echo "Once light curves are loaded, the data will be written to a parquet dataset with"
    echo "one partition for each mission."
    echo "To check which missions are done, set the 'parquet_dir' variable (copy/paste from script"
    echo "output), then use:"
    echo "    \$ ls -l \$parquet_dir"
    echo "You will see a directory for each mission that is complete."
    echo
    echo "---- Abort and Kill Jobs ----"
    echo "If a particular mission encounters problems and you need to abort its job, kill the process."
    echo "Set a 'pid' variable (copy/paste from script output), then use:"
    echo "    \$ kill \$pid"
    echo
    echo "If, at any point, you want to abort entirely and kill every process that was started"
    echo "by this script, call the script again using the flags '-r' (with same value as before)"
    echo "and '-k'. For example, if your run ID is 'my_run_id':"
    echo "    \$ ./$(basename $0) -r my_run_id -k"
    echo
    echo "---- Load the Light Curves (after jobs complete) ----"
    echo "Light curves will be written to a parquet dataset."
    echo "To load the data (in python, with pandas) set the 'parquet_dir' variable (copy/paste from"
    echo "script output), then use:"
    echo "    >>> df_lc = pd.read_parquet(parquet_dir)"
    echo

}

print_usage(){
    echo "---- Usage Examples ----"
    echo "  - Basic run with default options:"
    echo "    \$ ./$(basename $0) -r my_run_id "
    echo "  - Specify two papers to get sample objects from, and two missions to fetch light curves from:"
    echo "    \$ ./$(basename $0) -r run_two -l 'yang hon' -m 'gaia wise'"
    echo "  - Get ZTF light curves for 200 objects from the SDSS sample:"
    echo "    \$ ./$(basename $0) -r ztf -l SDSS -n 200 -m ztf"
    echo "  - If something went wrong with the 'ztf' run and you want to abort (kill all the jobs):"
    echo "    \$ ./$(basename $0) -r ztf -k"
    echo
    echo "---- Available Flags ----"
    echo "Flags that require a value (defaults in parentheses):"
    echo "    -r : ID for the run. Used to label the output directory. There is no default."
    echo "         A value is required. No spaces or special characters."
    echo "    -l ('yang SDSS') : Space-separated list of literature/paper names from which to build the object sample."
    echo "    -c (true) : whether to consolidate_nearby_objects"
    echo "    -g ('{"SDSS": {"num": 10}}') : json string representing dicts with keyword arguments."
    echo "    -m ('gaia heasarc icecube wise ztf') : Space-separated list of missions from which to load light curves."
    echo "    -o (object_sample.ecsv) : File name storing the object sample."
    echo "    -p (lightcurves.parquet) : Directory name storing the light-curve data."
    echo "Flags to be used without a value:"
    echo "    -h : Print this help message."
    echo "    -i : Print instructions on monitoring a run and loading the output."
    echo "    -k : Abort the run. Kill all processes started by the run with this ID ('-r')."
    echo
    echo "---- Instructions for Monitoring a Run ----"
    print_help_run_instructions
}

# ---- Set variable defaults.
literature_names=(yang SDSS)
literature_names_all=(green hon lamassa lopeznavas lyu macleod16 macleod19 ruan papers sheng SDSS TDE yang)
consolidate_nearby_objects=true
get_sample_kwargs='{"SDSS": {"num": 10}}'
mission_names=(gaia heasarc icecube wise ztf)
mission_names_core=(gaia heasarc icecube wise ztf)
mission_names_all=(gaia heasarc hcv icecube panstarrs tess_kepler wise ztf)
sample_filename=object_sample.ecsv
use_existing_sample=false
parquet_dataset_name=lightcurves.parquet
kill_all_processes=false

# ---- Set variables that were passed in as script arguments.
# info about getopts: https://www.computerhope.com/unix/bash/getopts.htm#examples
while getopts r:l:c:g:m:o:u:p:hik flag; do
    case $flag in
        r) run_id=$OPTARG;;
        l) literature_names=("$OPTARG")
            if [ "${literature_names[0]}" == "all" ]; then
                literature_names=("${literature_names_all[@]}")
            fi
            ;;
        c) consolidate_nearby_objects=$OPTARG;;
        g) get_sample_kwargs=$OPTARG;;
        m) mission_names=("$OPTARG")
            if [ "${mission_names[0]}" == "all" ]; then
                mission_names=("${mission_names_all[@]}")
            fi
            if [ "${mission_names[0]}" == "core" ]; then
                mission_names=("${mission_names_core[@]}")
            fi
            ;;
        o) sample_filename=$OPTARG;;
        u) use_existing_sample=$OPTARG;;
        p) parquet_dataset_name=$OPTARG;;
        h) print_usage
            exit 0
            ;;
        i) print_run_instructions
            exit 0
            ;;
        k) kill_all_processes=true;;
        ?) print_help_me
            exit 1
            ;;
      esac
done

# If a run_id was not supplied, exit.
if [ -z ${run_id+x} ]; then
    echo "./$(basename $0): missing required option -- r"
    print_help_me
    exit 1
fi

# ---- Construct file paths.
basedir="output/lightcurves-${run_id}"
sample_filepath="${basedir}/${sample_filename}"
parquet_dataset_path="${basedir}/${parquet_dataset_name}"
logsdir="${basedir}/logs"
mkdir -p $logsdir
mylogfile="${logsdir}/$(basename $0).log"
logfiles=("$mylogfile")

{  # we will tee the output of everything below here to $mylogfile

# ---- If the user has requested to kill processes, do it and then exit.
if [ $kill_all_processes == true ]; then 
    kill_all_pids $run_id $logsdir $mylogfile
    exit 0
fi

# ---- Report basic info about the run.
echo "*********************************************************************"
echo "**                          Run starting.                          **"
echo "**                                                                 **"
echo
echo "run_id=${run_id}"
echo "output_dir=${basedir}"
echo "This script output will also be written to logfile=${mylogfile}"

# ---- Do the run. ---- #

# ---- 1: Run job to get the object sample, if needed. Wait for it to finish.
if [ -f $sample_filepath ] && [ $use_existing_sample == true ]; then
    echo "Using existing object sample at: ${sample_filepath}"
else
    logfile="${logsdir}/get_sample.log"
    logfiles+=("$logfile")
    echo
    echo "Build sample is starting. logfile=${logfile}"
    python code_src/run_light_curve_generator.py --build sample \
        --literature_names ${literature_names[@]} \
        --consolidate_nearby_objects $consolidate_nearby_objects \
        --get_sample_kwargs "${get_sample_kwargs}" \
        --sample_filename $sample_filepath \
        &> ${logfile}
    echo "Build sample is done. Printing the log for convenience:"
    print_logs $logfile
fi

# ---- 2: Start the jobs to fetch the light curves in the background. Do not wait for them to finish.
echo
echo "Mission archive calls are starting."
for mission in ${mission_names[@]}; do
    logfile="${logsdir}/${mission}.log"
    logfiles+=("$logfile")
    nohup python code_src/run_light_curve_generator.py --build lightcurves \
        --mission ${mission} \
        --sample_filename ${sample_filepath} \
        --parquet_dataset_name ${parquet_dataset_path} \
        > ${logfile} 2>&1 &
    echo "[pid=${!}] ${mission} started. logfile=${logfile}"
done

# ---- 3: Print some instructions for the user, then exit.
echo
echo "Light curves are being loaded in background processes. PIDs are listed above."
echo "Once loaded, data will be written to a parquet dataset. parquet_dir=${parquet_dataset_path}"
print_help_run_instructions
echo
echo "**                       Main script exiting.                       **"
echo "**           Jobs may continue running in the background.           **"
echo "**********************************************************************"
} | tee -a $mylogfile