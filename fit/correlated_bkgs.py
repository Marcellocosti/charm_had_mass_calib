'''
    Normalizations are given for the fit function to use a signal PDF and
    template from hMassTotalCorrBkgs multiplied by a common normalization 
    constant.
'''

import pandas as pd
import uproot
import numpy as np
import ROOT
import os
import sys
import argparse
import yaml
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from ROOT import TFile, RooRealVar, RooDataSet, RooArgSet, RooKeysPdf, TFile, TH1F


final_states = {
    "DplusToPiKPi": {
      "flag_mc_rec": 1,
    },
    "DplusToPiKK": {
      "flag_mc_rec": 4,
    },
    "DsToPiKK": {
      "flag_mc_rec": 5,
    },
}



def fill_smooth_histo(df, histo, n_points_for_sample, n_points_for_kde):

    # Define the RooDataset corresponding to histogram range
    x_min = histo.GetXaxis().GetXmin()
    x_max = histo.GetXaxis().GetXmax()
    x = RooRealVar("x", "x", x_min, x_max)
    data = RooDataSet("data", "data", RooArgSet(x))

    # Fill it from DataFrame
    for i_val, val in enumerate(df['fM']):
        if i_val > n_points_for_kde:
            break
        x.setVal(val)
        data.add(RooArgSet(x))

    # Build a RooKeysPdf (kernel smoothing)
    keys_pdf = RooKeysPdf("keys", "keys", x, data, RooKeysPdf.NoMirror)
    generated = keys_pdf.generate(RooArgSet(x), n_points_for_sample)

    histo_smooth = histo.Clone(f"{histo.GetName()}")
    histo_smooth.Reset("ICESM")
    for i in range(int(generated.numEntries())):
        val = generated.get(i).getRealValue("x")
        histo_smooth.Fill(val)

    histo_smooth.Scale(len(df) / histo_smooth.Integral())
    return histo_smooth


def shift_templs(cfg_corrbkgs, cutset_sel_df, pt_min, pt_max):
    # pt-differential mass shifts
    # Copy the dataframe to avoid modifying the original one
    df = cutset_sel_df.copy(deep=True)
    
    mass_shift = 0.
    if isinstance(cfg_corrbkgs["shift_mass"], float):
        print(f"Applying constant mass shift of {cfg_corrbkgs['shift_mass']} GeV/c^2")
        mass_shift = cfg_corrbkgs["shift_mass"]
    else:
        print(f"Taking mass shifts from {cfg_corrbkgs['shift_mass']}")
        shifts_file = ROOT.TFile(cfg_corrbkgs['shift_mass'], "READ")
        shifts_histo = shifts_file.Get("delta_mean_data_mc")
        for i_bin in range(1, shifts_histo.GetNbinsX()+1):
            bin_center = shifts_histo.GetBinCenter(i_bin)
            if (bin_center > pt_min and bin_center < pt_max):
                mass_shift = shifts_histo.GetBinContent(i_bin)
                break
        shifts_histo.SetDirectory(0)
        shifts_file.Close()

    print(f"Shifting mass by {mass_shift} GeV/c^2")
    df.loc[:, "fM"] = df["fM"] + mass_shift
    return df


def smear_templs(cfg_corrbkgs, cutset_sel_df, pt_min, pt_max):
    # pt-differential mass smearing
    # Copy the dataframe to avoid modifying the original one
    df = cutset_sel_df.copy(deep=True)
    
    mass_smear = 0.
    if isinstance(cfg_corrbkgs["smear_mass"], float):
        sigma_smear = cfg_corrbkgs["smear_mass"]
    else:
        print(f"Taking mass smears from {cfg_corrbkgs['smear_mass']}")
        smear_file = ROOT.TFile(cfg_corrbkgs['smear_mass'], "READ")
        smear_histo = smear_file.Get("delta_sigma_data_mc")
        for i_bin in range(1, smear_histo.GetNbinsX()+1):
            bin_center = smear_histo.GetBinCenter(i_bin)
            if (bin_center > pt_min and bin_center < pt_max):
                sigma_smear = smear_histo.GetBinContent(i_bin)
                break
        smear_histo.SetDirectory(0)
        smear_file.Close()
        if sigma_smear > 0:
            mass_smear = np.random.normal(0.0, sigma_smear, size=len(df)).astype("float32")
            print(f"Smearing mass by sigma = {mass_smear[:10]} GeV/c^2")
            df.loc[:, "fM"] = df["fM"] + mass_smear
        else:
            print(f"Mass smearing value is {sigma_smear}, no smearing applied.")
    return df


def produce_chn_corrbkg(cfg_corrbkgs, df, outfile, chn_dir, templ_type='raw'):

    outfile.mkdir(f'{chn_dir}/{templ_type}')
    outfile.cd(f'{chn_dir}/{templ_type}')

    histo_mass = TH1F("hMass", "hMass", 700, 1.6, 2.3)
    treeFrac = ROOT.TTree("treeFrac", "treeFrac")
    treeMass = ROOT.TTree("treeMass", "treeMass")

    # Use arrays for branches
    fM_mass = np.zeros(1, dtype=np.float32)
    fM_frac = np.zeros(1, dtype=np.float32)
    fPt = np.zeros(1, dtype=np.float32)
    fCentrality = np.zeros(1, dtype=np.float32)
    fMlScore0 = np.zeros(1, dtype=np.float32)
    fMlScore1 = np.zeros(1, dtype=np.float32)

    treeMass.Branch("fM", fM_mass, "fM/F")
    treeFrac.Branch("fM", fM_frac, "fM/F")
    treeFrac.Branch("fPt", fPt, "fPt/F")
    treeFrac.Branch("fCentrality", fCentrality, "fCentrality/F")
    treeFrac.Branch("fMlScore0", fMlScore0, "fMlScore0/F")
    treeFrac.Branch("fMlScore1", fMlScore1, "fMlScore1/F")

    # Loop only once over DataFrame length
    mass_array = df['fM'].to_numpy(dtype=np.float32)
    pt_array = df['fPt'].to_numpy(dtype=np.float32)
    centrality_array = df['fCentrality'].to_numpy(dtype=np.float32)
    score0_array = df['fMlScore0'].to_numpy(dtype=np.float32)
    score1_array = df['fMlScore1'].to_numpy(dtype=np.float32)

    n_entries = len(df)
    for i in range(n_entries):
        histo_mass.Fill(mass_array[i])
        fM_mass[0] = mass_array[i]
        fM_frac[0] = mass_array[i]
        fPt[0] = pt_array[i]
        fCentrality[0] = centrality_array[i]
        fMlScore0[0] = score0_array[i]
        fMlScore1[0] = score1_array[i]
        treeFrac.Fill()
        treeMass.Fill()

    # Smoothed histogram
    histo_mass_smooth = histo_mass.Clone("hMassSmooth")
    histo_mass_smooth.Reset("ICESM")
    histo_mass_smooth = fill_smooth_histo(df, histo_mass_smooth,
                                          cfg_corrbkgs['n_smooth_points'],
                                          cfg_corrbkgs['n_points_for_kde'])
    histo_mass_smooth.Smooth(100)

    histo_mass.Write('hMassRaw')
    histo_mass_smooth.Write('hMassSmooth')
    treeFrac.Write('treeFracMassScoresBkgFD')
    treeMass.Write('treeMass')

    return histo_mass


def produce_corr_bkgs_templs(cfg, sel_cfg, cent_diff_df, pt_bins, bdt_sel_bkg, bdt_sel_nonprompt):

    # Precompute final-state masks for all entries
    decay_masks = {}
    for fin_state, info in final_states.items():
        decay_masks[fin_state] = (abs(cent_diff_df["fFlagMcMatchRec"]) == info["flag_mc_rec"])

    out_dir = os.path.dirname(cfg['outfile'])
    os.makedirs(out_dir, exist_ok=True)
    outfile = TFile(f"{out_dir}/templs_cent_{sel_cfg['centrality'][0]}_{sel_cfg['centrality'][1]}.root", "RECREATE")

    # Loop over pt bins
    print(f"length of dataframe: {len(cent_diff_df)}")
    for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
        print(f"\nProducing correlated background templates for pt interval: [{pt_min}, {pt_max}) GeV/c")
        pt_key = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        pt_bdt_mask = (cent_diff_df.fPt >= pt_min) & (cent_diff_df.fPt < pt_max) & \
                      (cent_diff_df.fMlScore0 <= bdt_sel_bkg[pt_bins.index(pt_min)]) & \
                      (cent_diff_df.fMlScore1 >= bdt_sel_nonprompt[pt_bins.index(pt_min)])
        df_pt = cent_diff_df[pt_bdt_mask].reset_index(drop=True)  # pt-selected DataFrame

        # Loop over final states using precomputed masks
        for fin_state, decay_mask_full in decay_masks.items():

            decay_pt_mask = decay_mask_full[pt_bdt_mask].reset_index(drop=True)

            # Apply pt mask
            n_candidates = decay_pt_mask.sum()
            if n_candidates <= cfg.get("min_entries", 0):
                print(f"----> No candidates for final state: {fin_state}!")
                continue

            print(f"Found {n_candidates} candidates for final state: {fin_state}")
            chn_dir = f"{pt_key}/{fin_state}"
            outfile.mkdir(chn_dir)
            outfile.cd(chn_dir)

            # Produce correlated backgrounds in a single pass per variant
            histo_mass = produce_chn_corrbkg(cfg, df_pt[decay_pt_mask], outfile, chn_dir, templ_type='raw')

            # Centrality histogram
            histo_cent = TH1F("hCentrality", "hCentrality", 101, -0.5, 100.5)
            histo_score_bkg = TH1F("hScoreBkg", "hScoreBkg", 1000, 0, 1)
            histo_score_fd = TH1F("hScoreFD", "hScoreFD", 1000, 0, 1)
            for row in df_pt[decay_pt_mask].itertuples(index=False):
                histo_cent.Fill(row.fCentrality)
                histo_score_bkg.Fill(row.fMlScore0)
                histo_score_fd.Fill(row.fMlScore1)

            # Scale histogram
            outfile.cd(chn_dir)
            histo_mass.Scale(1.0)
            histo_mass.Write('hMassScaled')
            histo_cent.Write('hCentrality')
            histo_score_bkg.Write('hScoreBkg')
            histo_score_fd.Write('hScoreFD')

    outfile.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Producing correlated backgrounds templates")
    tables = [[] for _ in config["table_names"]]
    for file_name in config["input_files"]:
        print(f"Reading file {file_name} for correlated bkg template production")
        with uproot.open(file_name) as f:
            for table_name, table_list, table_cols_to_keep in zip(
                config["table_names"],
                tables,
                config["table_cols_to_keep"]
            ):
                for key in f.keys():
                    if table_name in key:
                        dfData = f[key].arrays(table_cols_to_keep, library='pd')
                        table_list.append(dfData)

    # Now concatenate per table
    full_dfs = [
        pd.concat(table_list, ignore_index=True)
        for table_list in tables
    ]
    full_df = pd.concat(full_dfs, axis=1)

    # Print unique fCentrality values for debugging
    unique_centralities = full_df['fCentrality'].unique()

    for selection in config["selections"]:
        cent_diff_df = full_df.query(f"fCentrality >= {selection['centrality'][0]} and fCentrality < {selection['centrality'][1]}")
        produce_corr_bkgs_templs(config,
                                 selection,
                                 cent_diff_df,
                                 selection["pt_intervals"],
                                 selection.get("bdt_sel_bkg", [1.]*len(selection["pt_intervals"])),
                                 selection.get("bdt_sel_nonprompt", [0.]*len(selection["pt_intervals"]))
                                )
