This project can be used to evaluate anonymity algorithms. 
1. You have to specify the settings for the anonymity algorithms in the "evaluation_config.py" file. You have to insert the absolut path to the evaluation folder in the "evaluation_config.py" file. 
If you want to use the DP-Star algorithm and use the script to sync the config-files, you also have to insert the absolut path to the DP-Star project. Otherwise, you need to change the settings in the config file of the DP-Star project manually.
2. You have to state the name of the synthetic algorithm in the "evaluation_config.py" file.
3. In the "evaluation_config.py" file you can state whether you want to generate similarity measures for both algorithms or just for one
4. You have to put the raw dataset and the tesselation file in "/data/raw_data" as a csv file. 
5. Then you have to create the synthetic datasets.
You can do that manually and store the files in "/data/dpstar_synthetic_datasets", or use the "synthetic_dataset_config.py" script and then run the DP-Star algorithm. The datasets will then be stored in the given folder automatically. 
6. If you used the DP-Star Package, remember to use the "dpstar_for_dpmob.py" script to modify the datasets so they can be used in the mobility report package.
7. Run the "dpstar_reports_gen.py" script.
8. The similarity measure results will be in the "evaluation_results" folder in the "evaluation_database.csv" file


Plotting: 
1. State the settings of the runs you want to plot at the top of the "plot_similarity_measures.py" file
2. Run the script
3. Plots will be shown as pop-ups and stored in the "plots" folder inside the "evaluation_results" folder
