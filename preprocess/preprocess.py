'''
Usage: python3 preprocess.py config.yml [-w WORKERS]
'''
import os
import sys
import yaml
from ROOT import TFile
import argparse
import concurrent.futures
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{script_dir}/")
sys.path.append(f"{script_dir}/../utils/")
from utils import logger
from data_model import get_sparse_dict
from itertools import product
from pathlib import Path


def get_input_paths(year_input):

    input_cfg = None
    for cent_class, cent_file in year_input.items():
        cent_class = [float(x) for x in cent_class.split('_')]
        if centrality[0] >= cent_class[0] and centrality[1] <= cent_class[1]:
            input_cfg = cent_file
            break

    if input_cfg is None:
        return None

    if isinstance(input_cfg, str) and input_cfg.endswith(".root"):
        return [input_cfg]

    if isinstance(input_cfg, str) and input_cfg.endswith(".txt"):
        # Read all lines in a list and return
        with open(input_cfg, "r") as f:
            file_paths = [line.strip() for line in f if line.strip().endswith(".root")]
        return file_paths

    if isinstance(input_cfg, str):          # All .root files in a directory
        dir_path = input_cfg
        file_paths = [f"{dir_path}/{file}" for file in os.listdir(dir_path) \
                      if file.endswith(".root") and "AnalysisResults" in file]
        return file_paths

    elif isinstance(input_cfg, list):       # List of file paths
        file_paths = input_cfg
        return file_paths

    else:
        logger("Invalid type for 'files' in configuration. Must be a string (directory) or list of file paths.", "ERROR")
        sys.exit(1)


def process_sparse(hadron, cent_class, i_file, infile_path, out_dir, pt_bins, occ_min, occ_max, bdt_sels_bkg, bdt_sels_prompt, bdt_sels_nonprompt, pt_intervals):
    """
    Process a single sparse from an input file for all pt bins according to the configuration.

    Args:
        hadron (str): Type of D meson ('Dzero', 'Dplus', 'Ds')
        cent_class (int): centrality class
        i_file (int): index of the input file
        infile (TFile): input ROOT file
        sel_cfg (dict): full configuration dictionary
        sparse_cfg (dict): sparse configuration dictionary
        out_dir (str): output directory for pre-processed files
        occ_pt_combinations (list): list of occupancy and pt combinations
        bdt_sels (list): list of BDT bkg score selections
    """

    infile = TFile.Open(infile_path, 'read')
    logger(f'Processing file {i_file}, {infile.GetName()} for hadron {hadron}', level='INFO')
    sparse_dict = get_sparse_dict(hadron)
    sparse = infile.Get(sparse_dict['Path'])

    # Apply centrality cut if axis is available
    logger(f"Applying cent cut to sparse {sparse} with value {cent_class[0]} -- {cent_class[1]}", "INFO")
    sparse.GetAxis(sparse_dict['Cent']).SetRangeUser(cent_class[0], cent_class[1])
    sparse.GetAxis(sparse_dict['Occ']).SetRangeUser(occ_min, occ_max)

    out_file_dir = f"{out_dir}/jobs"
    os.makedirs(out_file_dir, exist_ok=True)
    out_file = TFile(f'{out_file_dir}/AnalysisResults_{i_file}.root', 'recreate')

    for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
        # Retrieve pt bin idx
        if bdt_sels_bkg is not None:
            bkg_max_cut = bdt_sels_bkg[pt_intervals.index(pt_min)]
            sparse.GetAxis(sparse_dict['ScoreBkg']).SetRangeUser(0, bkg_max_cut)
        if bdt_sels_prompt is not None:
            prompt_max_cut = bdt_sels_prompt[pt_intervals.index(pt_min)]
            sparse.GetAxis(sparse_dict['ScorePrompt']).SetRangeUser(0, prompt_max_cut)
        if bdt_sels_nonprompt is not None:
            nonprompt_max_cut = bdt_sels_nonprompt[pt_intervals.index(pt_min)]
            sparse.GetAxis(sparse_dict['ScoreFD']).SetRangeUser(0, nonprompt_max_cut)
        sparse.GetAxis(sparse_dict['Pt']).SetRangeUser(pt_min, pt_max)
        proj_sparse = sparse.Projection(sparse_dict['ScoreBkg'], sparse_dict['Mass'], 'O')
        proj_sparse.SetName(f"h2D_{hadron}_Occ_{occ_min}_{occ_max}_Pt_{pt_min}_{pt_max}")
        proj_sparse.Write()

    infile.Close()
    out_file.Close()
    logger(f'----> Finished processing file {i_file}, sparse: {sparse_dict["Path"]}\n', "INFO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('config', metavar='text', default='config.yml', help='configuration file')
    parser.add_argument("--workers", "-w", type=int, default=1, help="number of workers")
    args = parser.parse_args()

    with open(args.config, 'r') as cfg_pre:
        full_cfg = yaml.safe_load(cfg_pre)

    # Loop over the charm hadrons
    output_dir = full_cfg['PreprocessOutDir']
    for hadron_cfg in full_cfg['CharmHadrons']:
        print("\n\n\n")
        logger(f"##### Processing {hadron_cfg['Name']} #####", "WARNING")
        output_dir_hadron = f"{output_dir}/{hadron_cfg['Name']}"
        os.makedirs(output_dir_hadron, exist_ok=True)

        # Loop over the selections
        for sel_cfg in hadron_cfg['Selections']:
            centrality = [float(x) for x in sel_cfg['Centrality'].split('_')] \
                         if isinstance(sel_cfg['Centrality'], str) else [sel_cfg['Centrality']]

            # Compute all combinations of occupancy and pt intervals
            for i_occ_bin, (occ_min, occ_max) in enumerate(zip(sel_cfg["Occupancy"][:-1], sel_cfg["Occupancy"][1:])):
                logger(f"Processing centrality {centrality}, occupancy {occ_min} -- {occ_max}", "INFO")
                # Loop over the different years
                for year, files in hadron_cfg['Inputs'].items():
                    file_paths = get_input_paths(files)

                    # Retrieve bdt sels, can be year-dependent and lists 
                    # for when occupancy diff studies are conducted
                    bdt_sels_bkg, bdt_sels_prompt, bdt_sels_nonprompt = None, None, None
                    if sel_cfg.get('BdtSelsBkg'):
                        bdt_sels_bkg = sel_cfg['BdtSelsBkg']
                        if isinstance(bdt_sels_bkg, list):
                            bdt_sels_bkg = bdt_sels_bkg[i_occ_bin]
                        else:
                            bdt_sels_bkg = bdt_sels_bkg[year]
                    if sel_cfg.get('BdtSelsPrompt'):
                        bdt_sels_prompt = sel_cfg['BdtSelsPrompt']
                        if isinstance(bdt_sels_prompt, list):
                            bdt_sels_prompt = bdt_sels_prompt[i_occ_bin]
                        else:
                            bdt_sels_prompt = bdt_sels_prompt[year]
                    if sel_cfg.get('BdtSelsNonPrompt'):
                        bdt_sels_nonprompt = sel_cfg['BdtSelsNonPrompt']
                        if isinstance(bdt_sels_nonprompt, list):
                            bdt_sels_nonprompt = bdt_sels_nonprompt[i_occ_bin]
                        else:
                            bdt_sels_nonprompt = bdt_sels_nonprompt[year]

                    if file_paths is None:
                        logger(f"No matching centrality found for input, skipping!", "ERROR")
                        continue

                    out_dir = f"{output_dir_hadron}/cent_{int(centrality[0])}_{int(centrality[1])}_occ_{int(occ_min)}_{int(occ_max)}/{year}"
                    os.makedirs(out_dir, exist_ok=True)
                    logger(f"##### Skimming centrality {centrality}, year {year} #####", "WARNING")
                    with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
                        tasks_sparses = [executor.submit(process_sparse, hadron_cfg['Name'], centrality,
                                                         i_file, file, out_dir, sel_cfg["PtIntervals"], occ_min, occ_max,
                                                         bdt_sels_bkg, bdt_sels_prompt, bdt_sels_nonprompt,
                                                         sel_cfg["PtIntervals"]) 
                                         for i_file, file in enumerate(file_paths)]
                    # Throw exceptions
                    for task in tasks_sparses:
                        try:
                            task.result()
                        except Exception as e:
                            logger(f"Error in processing sparse: {e}", "ERROR")

                    job_dir = Path(out_dir) / "jobs"
                    paths_file = Path(out_dir) / "jobs_file_paths.txt"
                    paths_file.write_text("\n".join(str(p) for p in job_dir.glob("*.root")))

                    # Merge all the output files
                    os.system(f"hadd -f -v 1 {out_dir}/MassDistributions.root @{out_dir}/jobs_file_paths.txt")
                    logger(f"Finished processing {out_dir}\n\n", "INFO")
