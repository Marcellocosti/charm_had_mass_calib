import os
import pandas as pd
from ROOT import TFile

MASSES_PDG = {"D0": 1.86484, "DplusKPiPi": 1.86955, "DplusKKPi": 1.86955, "Ds": 1.96835, "Lc": 2.28646}
MASSES_PDG_UNCS = {"D0": 0.00004, "DplusKPiPi": 0.00027, "DplusKKPi": 0.00027, "Ds": 0.00007, "Lc": 0.00014}

def compute_shift(mass_meas, mass_meas_unc, particle):
    """
    Compute the mass shift and its uncertainty.
    """
    mass_pdg = MASSES_PDG[particle]
    mass_pdg_unc = MASSES_PDG_UNCS[particle]
    shift = mass_meas - mass_pdg
    shift_unc = (mass_meas_unc**2 + mass_pdg_unc**2)**0.5
    return shift, shift_unc

def get_d0_lc_results(part_name, path, years, debug=False):
    results = {}
    for year_folder in years:
        results[year_folder] = {}
        for file in os.listdir(f"{path}/{part_name}/{year_folder}"):
            if not file.endswith(".root"):
                continue

            results_file_path = f"{path}/{part_name}/{year_folder}/{file}"
            results_file = TFile(results_file_path, "READ")
            hist_raw_yields = results_file.Get("hRawYields")
            hist_means = results_file.Get("hRawYieldsMean")
            hist_sigmas = results_file.Get("hRawYieldsSigma")

            file_name = os.path.basename(results_file_path).format(part_name=part_name, year_folder=year_folder)
            cent_class = file_name.split("cent_")[1].replace(".root", '')
            results[year_folder][cent_class] = {}
            for i in range(1, hist_means.GetNbinsX() + 1):
                pt_label = f"pt_{int(hist_means.GetXaxis().GetBinLowEdge(i)*10)}_{int(hist_means.GetXaxis().GetBinUpEdge(i)*10)}"
                results[year_folder][cent_class][pt_label] = {}
                results[year_folder][cent_class][pt_label]["pt_min"] = hist_means.GetXaxis().GetBinLowEdge(i)
                results[year_folder][cent_class][pt_label]["pt_max"] = hist_means.GetXaxis().GetBinUpEdge(i)
                results[year_folder][cent_class][pt_label]["raw_yields"] = hist_raw_yields.GetBinContent(i)
                results[year_folder][cent_class][pt_label]["raw_yields_unc"] = hist_raw_yields.GetBinError(i)
                results[year_folder][cent_class][pt_label]["mean"] = hist_means.GetBinContent(i)
                results[year_folder][cent_class][pt_label]["mean_unc"] = hist_means.GetBinError(i)
                results[year_folder][cent_class][pt_label]["sigma"] = hist_sigmas.GetBinContent(i)
                results[year_folder][cent_class][pt_label]["sigma_unc"] = hist_sigmas.GetBinError(i)
                results[year_folder][cent_class][pt_label]["shift"], \
                results[year_folder][cent_class][pt_label]["shift_unc"] =  \
                    compute_shift(
                        hist_means.GetBinContent(i),
                        hist_means.GetBinError(i),
                        part_name
                    )

    if debug:
        print(f"{part_name} results:")
        for year, cent_classes in results.items():
            for cent_class, pt_bins in cent_classes.items():
                for pt_bin, values in pt_bins.items():
                    print(f"Year: {year}, Cent: {cent_class}, Pt bin: {pt_bin}, Mean: {values['mean']:.5f} ± {values['mean_unc']:.5f}, Sigma: {values['sigma']:.5f} ± {values['sigma_unc']:.5f}")

    return results


def get_dplus_ds_results(input_file, particle_map, debug=False):
    """
    Parameters
    ----------
    input_file : str
        Path to fit_results.parquet.

    particle_map : dict
        Maps particle name -> index in the parquet arrays.
        Example:
            {"DplusKPiPi": 0}
            {"Ds": 0, "DplusKKPi": 1}

    Returns
    -------
    dict
        results[particle][cent_class][pt_label]
    """

    results = {particle: {} for particle in particle_map}

    results_df = pd.read_parquet(input_file)

    for row in results_df.itertuples(index=False):

        cent_class = f"{row.cent_min_cfg}_{row.cent_max_cfg}"
        pt_label = f"pt_{int(row.pt_min_cfg*10)}_{int(row.pt_max_cfg*10)}"

        for particle, idx in particle_map.items():

            if cent_class not in results[particle]:
                results[particle][cent_class] = {}

            shift, shift_unc = compute_shift(
                row.mu[idx][0],
                row.mu[idx][1],
                particle,
            )

            results[particle][cent_class][pt_label] = {
                "pt_min": row.pt_min_cfg,
                "pt_max": row.pt_max_cfg,
                "raw_yields": row.raw_yields[idx][0],
                "raw_yields_unc": row.raw_yields[idx][1],
                "mean": row.mu[idx][0],
                "mean_unc": row.mu[idx][1],
                "sigma": row.sigma[idx][0],
                "sigma_unc": row.sigma[idx][1],
                "shift": shift,
                "shift_unc": shift_unc,
            }
            
    if debug:
        print("D+ ---> KPiPi results:")
        for year, cent_classes in results.items():
            for cent_class, pt_bins in cent_classes.items():
                for pt_bin, values in pt_bins.items():
                    print(f"Year: {year}, Cent: {cent_class}, Pt bin: {pt_bin}, Mean: {values['mean']:.5f} ± {values['mean_unc']:.5f}, Sigma: {values['sigma']:.5f} ± {values['sigma_unc']:.5f}")


    return results