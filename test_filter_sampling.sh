# #!/bin/bash

# # if environment variable DATASET_DIR is not specified
# # export DATASET_DIR=/fred/oz016/datasets/
# # env DATASET_DIR=/fred/oz016/datasets/ bash generate_datasets.sh
# # then we set it to be a default path

# dataset_dir=${DATASET_DIR:-"<PATH TO OUTPUT DATASET DIR>"}

dataset_dir=${DATASET_DIR:-"data"}
echo "Saving datasets to ${dataset_dir}/"

folder="test_snr_generation"

# timer
SECONDS=0

# Sample Parameters from Priors

# generate sample parameters
python -m gwpe.parameters \
    -n 1000 \
    -d "${dataset_dir}/${folder}/" \
    -c gwpe/config_files/intrinsics.ini \
    -c gwpe/config_files/extrinsics.ini \
    --overwrite \
    --metadata \
    --verbose

# typically we would estimate a PSD via welch estimate on real strain
# due to data location issues we include a sample PSD for testing
# copy sample PSD to dataset partition locations
for partition in ${folder}
do
    for ifo in "H1" "L1"
    do
        cp -r "PSD" "${dataset_dir}/${partition}"
    done

done

## Waveforms
# run waveform generation script for training data
python -m gwpe.waveforms \
    -d "${dataset_dir}/${folder}/" \
    -s gwpe/config_files/static_args.ini \
    --psd_dir "${dataset_dir}/${folder}/PSD/" \
    --ifos "H1" "L1" \
    --ref_ifo "H1" \
    --add_noise \
    --whiten \
    --highpass \
    --overwrite \
    --verbose \
    --validate \
    --metadata \
    --chunk_size 5 \
    --workers 10
#    --gaussian \

# generate filters for all samples
python -m gwpe.filter_parameters \
    -n 10 \
    -b \
    -d "${dataset_dir}/${folder}/" \
    -s 'gwpe/config_files/static_args.ini' \
    -t 'gwpe/config_files/template_params.txt' \
    -v

# print out size of datasets
for partition in ${folder}
do
    echo "$(du -sh ${dataset_dir}/${partition}/)"
done

# print out runtime
if (( $SECONDS > 3600 )) ; then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)."
elif (( $SECONDS > 60 )) ; then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)."
else
    echo "Completed in $SECONDS seconds."
fi
