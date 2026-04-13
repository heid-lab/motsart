PROJECT_NAME='finetune_noise_1_TS'

python -m motsart.learning.data_pkl_to_results \
    --fetch_cluster your_cluster \
    --fetch_pkl_path /home/leonard.galustian/projects/goflow/reaction_analysis/${PROJECT_NAME}/samples_all.pkl \
    --local_fetch_dir data/samples_musica/${PROJECT_NAME} \
    --local_results_folder results_goflow \
    --dest_cluster your_cluster \
    --dest_dir /data/results_goflow/${PROJECT_NAME}