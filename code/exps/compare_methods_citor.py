import argparse, os

from lib.CITOR import CITOR, load_python_object_encrypted
from lib.experiment import CompareMethods
import shutil, yaml, pandas


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def prep_citor_data(citor_data):
    # Specify that the development data is from the UMCG (the case for CITOR)
    citor_data.dev_data['ZIEKH']=3.0

    # fix naming difference between dev and val data # rename and convert all grades > 3 to 3
    citor_data.dev_data['DYSFAGIE_UMCGshortv2_BSL'] = citor_data.dev_data.apply(lambda row: row['DYSFAGIE_CTCAEv4_BSL'] if (pandas.isna(row['DYSFAGIE_CTCAEv4_BSL']) or row['DYSFAGIE_CTCAEv4_BSL'] in [1.0, 2.0, 3.0]) else 3.0, axis=1)
    citor_data.dev_data['DYSFAGIE_UMCGshortv2_M06']= citor_data.dev_data.apply(lambda row: row['DYSFAGIE_CTCAEv4_M06'] if (pandas.isna(row['DYSFAGIE_CTCAEv4_M06']) or row['DYSFAGIE_CTCAEv4_M06'] in [1.0, 2.0, 3.0]) else 3.0, axis=1)

    # TCAT: Tx, T0, Tis, T1, T2, T3, T4, T4a, T4b = 1, 2, 3, 4, 5, 6, 7, 8, 9
    Tmap = {1:2, 2:4, 3:5, 4:6, 5:7, 6:3, 7:1, 8:8, 9:9}
    citor_data.dev_data['TCAT'] = citor_data.dev_data['TSTAD_DEF']
    citor_data.val_data['TCAT'] = [Tmap[v] for v in citor_data.val_data['TSTAD_DEF'].values]


    # NCAT: Nx, N0, N1, N2a, N2b, N2c, N3 = 1, 2, 3, 4, 5, 6, 7
    Nmap = {1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:1}
    citor_data.dev_data['NCAT'] = citor_data.dev_data['NSTAD_DEF']
    citor_data.val_data['NCAT'] = [Nmap[v] for v in citor_data.val_data['NSTAD_DEF'].values]


    # Select all variables that are shared between the development and validation data
    shared_vars = [k for k in citor_data.val_data.keys() if k in citor_data.dev_data.keys()]
    citor_df = pandas.concat([citor_data.dev_data.filter(items=shared_vars), citor_data.val_data.filter(items=shared_vars)], ignore_index=True)

    return  citor_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comparing some models.')
    parser.add_argument('-yaml', type=str,
                        help='Yaml configuration file containing all details for the experiment.')
    parser.add_argument('-bs', type=int, default=0,
                        help='The seed used in case of bootstrapping, default=0 (no bootstrap)')
    parser.add_argument('-pwd', type=str, default="password",
                        help='Password used to decrypt the dataset.')
    args = parser.parse_args()

    # Load the yaml config file to get the details about the experiment
    with open(args.yaml) as config_file:
        yaml_dict = yaml.full_load(config_file)

    # Load and decrypt the encrypted CITOR data using the given password
    citor_data = load_python_object_encrypted(yaml_dict['data_path'], args.pwd)
    citor_df = prep_citor_data(citor_data)

    # Make some output directories and metric and model files (to summarize the results from different bootstraps)
    general_out_folder=yaml_dict['output_dir'] + '/' + yaml_dict['exp_name']
    bs_out_folder=general_out_folder + '/bootraps/bs-' + str(args.bs) + '/'
    metric_file = general_out_folder + '/metrics.csv'
    train_metric_file = general_out_folder + '/train_metrics.csv'

    models_file = general_out_folder + '/models.csv'
    tex_metric_file = general_out_folder + '/metrics_table.tex'
    tex_train_metric_file = general_out_folder + '/train_metrics_table.tex'


    print('comparison...', yaml_dict['models'])

    comparison = CompareMethods(data=citor_df, out_folder=bs_out_folder, name=yaml_dict['exp_name'])

    comparison.conduct(metrics_file=metric_file, train_metrics_file=train_metric_file, models_file=models_file, yaml_dict=yaml_dict, bs=args.bs)

    shutil.copyfile(args.yaml, general_out_folder + '/config.yml')





