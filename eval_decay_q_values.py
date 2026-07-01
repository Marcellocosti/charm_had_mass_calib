import os
import numpy as np
import ROOT
from ROOT import TFile
from array import array
from ROOT import TMultiGraph, TGraphErrors
from utils.plot_utils import set_figure_style, make_canvas, make_legend, set_graph_style, set_plot_style, set_fit_style
from utils.load_utils import get_d0_lc_results, get_dplus_ds_results, MASSES_PDG

set_figure_style()

ROOT.TH1.AddDirectory(False)
ROOT.gErrorIgnoreLevel = ROOT.kError

BASEPATH = "/data/shared/CharmHadMassCalib"
os.makedirs(f"{BASEPATH}/final_plots", exist_ok=True)
os.makedirs(f"{BASEPATH}/final_plots/means", exist_ok=True)
os.makedirs(f"{BASEPATH}/final_plots/sigmas", exist_ok=True)
os.makedirs(f"{BASEPATH}/final_plots/shifts", exist_ok=True)
os.makedirs(f"{BASEPATH}/final_plots/calib_mass", exist_ok=True)


# Load inputs as dictionaries
YEARS = ["2023", "2024", "2025"]

d0_results = get_d0_lc_results("D0", "/data/shared/CharmHadMassCalib/outputs", YEARS, debug=True)
lc_results = get_d0_lc_results("Lc", "/data/shared/CharmHadMassCalib/outputs", YEARS, debug=True)

ds_results, dplus_kpipi_results, dplus_kkpi_results = {}, {}, {}
for year in YEARS:
    dplus_kpipi_results[year] = get_dplus_ds_results(f"/data/shared/CharmHadMassCalib/outputs/Dplus/{year}/fit_results.parquet",
                                                     {"DplusKPiPi": 0})["DplusKPiPi"]

    res = get_dplus_ds_results(f"/data/shared/CharmHadMassCalib/outputs/Ds/{year}/fit_results.parquet",
                               {"Ds": 0, "DplusKKPi": 1})
    ds_results[year] = res["Ds"]
    dplus_kkpi_results[year] = res["DplusKKPi"]


# Plot mean and sigma values for D0, D+, Ds, and Lc as a function of pt for each year and centrality class
RESULTS = {"D0": d0_results, "DplusKPiPi": dplus_kpipi_results, "DplusKKPi": dplus_kkpi_results, "Ds": ds_results, "Lc": lc_results}
MASSES_PDG_LINES = []
PARTICLE_NAMES = {
    "D0": "D^{0}",
    "DplusKPiPi": "D^{+}",
    "DplusKKPi": "D^{+}",
    "Ds": "D_{s}^{+}",
    "Lc": "#Lambda_{c}^{+}",
}

# Axis labels
AXIS_LABELS = {
    "mean": (
        lambda particle: f"M({PARTICLE_NAMES[particle]}) (GeV/#it{{c}}^{{2}})",
        lambda particle: MASSES_PDG[particle],
    ),
    "sigma": (
        lambda particle: f"#sigma({PARTICLE_NAMES[particle]}) (MeV/#it{{c}}^{{2}})",
        None,
    ),
    "shift": (
        lambda particle: f"M_{{meas}} - M_{{PDG}}({PARTICLE_NAMES[particle]}) (MeV/#it{{c}}^{{2}})",
        lambda particle: 0.,
    ),
}


def weighted_average_with_uncertainty(values, values_unc, weights, weights_unc):
    values = np.asarray(values, dtype=float)
    values_unc = np.asarray(values_unc, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights_unc = np.asarray(weights_unc, dtype=float)

    S = np.sum(weights)
    mean = np.sum(weights * values) / S

    var_values = np.sum((weights / S)**2 * values_unc**2)
    var_weights = np.sum(((values - mean) / S)**2 * weights_unc**2)

    unc = np.sqrt(var_values + var_weights)

    return mean, unc

def get_shifts_with_uncs(particle, results, year, cent_class, pt_bins):

    cent_min, cent_max = map(float, cent_class.split("_"))
    shifts, shifts_uncs = [], []

    for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):

        pt_label = f"pt_{pt_min}_{pt_max}"
        selected = []

        # Case 1: exact centrality and exact pT bin
        if cent_class in results[year] and pt_label in results[year][cent_class]:
            print(f"Exact match found for {particle} in centrality {cent_class}, pT bin {pt_label}, year {year}.")
            selected = [results[year][cent_class][pt_label]]

        else:
            # Determine which centrality classes contribute
            if cent_class in results[year]:
                cent_classes = [cent_class]
            else:
                cent_classes = [
                    cc for cc in results[year]
                    if (lambda cmin, cmax:
                        cmin >= cent_min and cmax <= cent_max)(
                            *map(float, cc.split("_"))
                        )
                ]

            if len(cent_classes) == 0:
                print(f"No centrality classes found for {particle} in centrality {cent_class}, year {year}. Skipping.")
                continue

            # Collect pT bins contained in requested interval
            pt_bins_sel = []
            for cc in cent_classes:
                for v in results[year][cc].values():
                    if pt_min <= v["pt_min"] and v["pt_max"] <= pt_max:
                        selected.append(v)
                        pt_bins_sel.append(f"pt_{v['pt_min']}_{v['pt_max']}")

            print(f"Averaging results for {particle} in centrality {cent_class}, pT bin {pt_label}, "
                  f"year {year} over centrality classes: {cent_classes} and pT bins: "
                  f"{pt_bins_sel}.")

        shift_avg, shift_unc = weighted_average_with_uncertainty(
            [v["shift"] for v in selected],
            [v["shift_unc"] for v in selected],
            [v["raw_yields"] for v in selected],
            [v["raw_yields_unc"] for v in selected],
        )

        shifts.append(shift_avg)
        shifts_uncs.append(shift_unc)

    return shifts, shifts_uncs

# Plots vs Q-value
MASS_PI        = 0.13957
MASS_K         = 0.49367
MASS_PROTON    = 0.93827
MASS_DEUTERON  = 1.87561
MASS_DPLUS     = 1.86965
MASS_DZERO     = 1.86483
MASS_DS        = 1.96834
MASS_LC        = 2.28646
MASS_CDEUTERON = 3.212

Q_VALUE_DPLUS_KPIPI    = MASS_DPLUS - MASS_K - 2*MASS_PI
Q_VALUE_DPLUS_KKPI     = MASS_DPLUS - 2*MASS_K - MASS_PI
Q_VALUE_DZERO_KPI      = MASS_DZERO - MASS_K - MASS_PI
Q_VALUE_DS_KKPI        = MASS_DS - 2*MASS_K - MASS_PI
Q_VALUE_LC_PKPIPI      = MASS_LC - MASS_PROTON - MASS_K - MASS_PI
Q_VALUE_CDEUTERON_DKPI = MASS_CDEUTERON - MASS_DEUTERON - MASS_K - MASS_PI

QVALUES = {"D0": Q_VALUE_DZERO_KPI, "DplusKPiPi": Q_VALUE_DPLUS_KPIPI, "DplusKKPi": Q_VALUE_DPLUS_KKPI, "Ds": Q_VALUE_DS_KKPI, "Lc": Q_VALUE_LC_PKPIPI}

print(f"Q-value for C-deuteron -> d K- pi+: {Q_VALUE_CDEUTERON_DKPI:.5f} GeV/c^2")
print(f"Q-value for Lambda_c -> p- K- pi+: {Q_VALUE_LC_PKPIPI:.5f} GeV/c^2")
print(f"Q-value for D+ -> K- K+ pi+: {Q_VALUE_DPLUS_KKPI:.5f} GeV/c^2")
print(f"Q-value for Ds -> K- K+ pi+: {Q_VALUE_DS_KKPI:.5f} GeV/c^2")
print(f"Q-value for D+ -> K- pi+ pi+: {Q_VALUE_DPLUS_KPIPI:.5f} GeV/c^2")
print(f"Q-value for D0 -> K- pi+: {Q_VALUE_DZERO_KPI:.5f} GeV/c^2")


# PLOTS OF MEAN, SIGMA, AND SHIFTS VS PT FOR DIFFERENT PARTICLES AND YEARS
out_file_vs_pt = TFile.Open(f"{BASEPATH}/final_plots/means_sigmas_shifts_vs_pt.root", "RECREATE")
for variable in ["mean", "sigma", "shift"]:
    for particle, results in RESULTS.items():
        for cent_class in next(iter(results.values())).keys():

            c = make_canvas(f"{variable}_{particle}_{cent_class}")
            leg = make_legend()

            mg = TMultiGraph()
            keep = []

            for year in YEARS:
                if cent_class not in results[year]:
                    continue

                x, ex, y, ey = [], [], [], []
                for v in results[year][cent_class].values():

                    x.append(0.5 * (v["pt_min"] + v["pt_max"]))
                    ex.append(0.5 * (v["pt_max"] - v["pt_min"]))

                    value = v[variable]
                    error = v[f"{variable}_unc"]

                    # Convert to MeV
                    if variable in ["sigma", "shift"]:
                        value *= 1000.
                        error *= 1000.

                    y.append(value)
                    ey.append(error)

                # Sort the points by x-value (pT) for proper plotting
                sorted_indices = np.argsort(x)
                x = [x[i] for i in sorted_indices]
                ex = [ex[i] for i in sorted_indices]
                y = [y[i] for i in sorted_indices]
                ey = [ey[i] for i in sorted_indices]

                g = TGraphErrors(len(x), array("d", x), array("d", y), array("d", ex), array("d", ey))
                set_graph_style(g, year)
                mg.Add(g)
                keep.append(g)
                leg.AddEntry(g, year, "lp")
                out_file_vs_pt.cd()
                g.SetDrawOption("PE")

                g.SetTitle(f"{variable}_vs_cent;#it{{p}}_{{T}} (GeV/#it{{c}});{variable} ({PARTICLE_NAMES[particle]})")
                g.Write(f"{variable}_{particle}_{cent_class}_{year}")

            set_plot_style(mg, xtitle="#it{p}_{T} (GeV/#it{c})", variable=variable,
                          labels=AXIS_LABELS, particle=particle, cent_class=cent_class)

            leg.Draw()

            c.SaveAs(f"{BASEPATH}/final_plots/{variable}s/{variable}_vs_pt_{particle}_{cent_class}.pdf")
out_file_vs_pt.Close()


### PLOTS OF MEAN, SIGMA, AND SHIFTS VS CENTRALITY FOR DIFFERENT PARTICLES AND YEARS
out_file_vs_cent = TFile.Open(f"{BASEPATH}/final_plots/means_sigmas_shifts_vs_cent.root", "RECREATE")
for variable in ["mean", "sigma", "shift"]:

    for particle, results in RESULTS.items():

        best_pt_bin = max(next(iter(next(iter(results.values())).values())).items(),
                          key=lambda kv: kv[1]["raw_yields"])[0]

        c = make_canvas(f"{variable}_cent_{particle}")
        leg = make_legend()

        mg = TMultiGraph()
        keep = []

        for year in YEARS:

            x, ex, y, ey = [], [], [], []
            for cent_class, pt_bins in results[year].items():

                if best_pt_bin not in pt_bins:
                    continue

                print(f"Using best pT bin {best_pt_bin} for {particle} in centrality {cent_class}, year {year}.")
                cmin, cmax = map(float, cent_class.split("_"))
                v = pt_bins[best_pt_bin]

                value = v[variable]
                error = v[f"{variable}_unc"]

                if variable in ["sigma", "shift"]:
                    value *= 1000.
                    error *= 1000.

                x.append(0.5 * (cmin + cmax))
                ex.append(0.5 * (cmax - cmin))
                y.append(value)
                ey.append(error)

            # Sort the points by centrality
            sorted_indices = np.argsort(x)
            x = np.array(x)[sorted_indices]
            ex = np.array(ex)[sorted_indices]
            y = np.array(y)[sorted_indices]
            ey = np.array(ey)[sorted_indices]

            g = TGraphErrors(len(x), array("d", x), array("d", y), array("d", ex), array("d", ey))
            set_graph_style(g, year)
            mg.Add(g)
            keep.append(g)
            leg.AddEntry(g, year, "lp")
            out_file_vs_cent.cd()
            g.SetDrawOption("PE")
            g.SetTitle(f"{variable}_vs_cent;Centrality (%);{variable} ({PARTICLE_NAMES[particle]})")
            g.Write(f"{variable}_{particle}_{best_pt_bin}_{year}")

        set_plot_style(mg, xtitle="Centrality (%)", variable=variable, labels=AXIS_LABELS,
                       particle=particle, pt_range=best_pt_bin.replace("pt_", "").split("_"))

        leg.Draw()

        c.SaveAs(f"{BASEPATH}/final_plots/{variable}s/{variable}_vs_cent_{particle}.pdf")
out_file_vs_cent.Close()


### PLOTS OF SHIFTS VS Q-VALUE FOR DIFFERENT PARTICLES AND YEARS
plot_cfgs = {
    "vs_pt_cent_0_20": {
        "pt_bins": [1, 2, 4, 6, 8, 12, 24],
        "cent_class": "0_20"
    }
}

out_file_vs_qvalue = TFile.Open(f"{BASEPATH}/final_plots/shifts_vs_qvalue.root", "RECREATE")
for plot_name, plot_cfg in plot_cfgs.items():

    pt_bins = plot_cfg["pt_bins"]
    cent_class = plot_cfg["cent_class"]

    shifts_dict = {}

    for particle, results in RESULTS.items():
        shifts_dict[particle] = {}

        for year in YEARS:
            shifts, uncs = get_shifts_with_uncs(particle, results, year, cent_class, pt_bins)
            shifts_dict[particle][year] = {
                "shifts": shifts,
                "uncs": uncs,
            }

    cd_extrap_vs_pt = {}
    lc_extrap_vs_pt = {}

    for ipt, (ptmin, ptmax) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):

        c = make_canvas(f"{plot_name}_{ipt}")
        leg = make_legend(x1=0.45, y1=0.80, x2=0.65, y2=0.92)

        mg = TMultiGraph()
        keep = []
        lin_fits = []
        fit_res = []
        cd_extrap = []
        lc_extrap = []

        for year in YEARS:

            x, ex, y, ey = [], [], [], []
            for particle in RESULTS:

                x.append(QVALUES[particle])
                ex.append(0.0)
                y.append(1000. * shifts_dict[particle][year]["shifts"][ipt])  # In MeV/c^2
                ey.append(1000. * shifts_dict[particle][year]["uncs"][ipt])   # In MeV/c^2

            # Sort the points by x-value (Q-value) for proper plotting
            sorted_indices = np.argsort(x)
            x = [x[i] for i in sorted_indices]
            ex = [ex[i] for i in sorted_indices]
            y = [y[i] for i in sorted_indices]
            ey = [ey[i] for i in sorted_indices]

            g = TGraphErrors(len(x), array("d", x), array("d", y), array("d", ex), array("d", ey))
            lin_fits.append(ROOT.TF1(f"lin_fit_{year}", "pol1", Q_VALUE_DPLUS_KKPI - 0.01, Q_VALUE_DZERO_KPI + 0.1))
            fit_res.append(g.Fit(lin_fits[-1], "R0S"))  # Fit with a linear function, suppress output
            set_graph_style(g, year)
            mg.Add(g)
            keep.append(g)
            leg.AddEntry(g, year, "lp")
            out_file_vs_qvalue.cd()
            g.SetDrawOption("PE")
            g.SetTitle(f"shifts_vs_qvalue_{plot_name}_{ptmin}_{ptmax}_{year};Q value (GeV/#it{{c}}^{{2}});Mass shift (MeV/#it{{c}}^{{2}})")
            g.Write(f"shifts_vs_qvalue_{plot_name}_{ptmin}_{ptmax}_{year}")

        set_plot_style(mg, xtitle="Q value (GeV/#it{c}^{2})", variable="shift",
                       labels=AXIS_LABELS, cent_class=cent_class, pt_range=(ptmin, ptmax))

        for fit, year in zip(lin_fits, YEARS):
            set_fit_style(fit, year)
            fit.DrawClone("same")

            fit.SetRange(Q_VALUE_CDEUTERON_DKPI - 0.01, Q_VALUE_DZERO_KPI + 0.1)
            fit.SetLineStyle(ROOT.kDashed)
            fit.DrawClone("same")

            xarr = array("d", [Q_VALUE_CDEUTERON_DKPI, Q_VALUE_LC_PKPIPI])
            cierr = array("d", [0.0, 0.0])
            fit_res[-1].GetConfidenceIntervals(2, 1, 1, xarr, cierr, 0.683, False)
            cd_err = cierr[0]
            lc_err = cierr[1]

            cd_extrap.append((fit.Eval(Q_VALUE_CDEUTERON_DKPI), cd_err))
            lc_extrap.append((fit.Eval(Q_VALUE_LC_PKPIPI), lc_err))

        frame = c.GetFrame()
        cd_line = ROOT.TLine(Q_VALUE_CDEUTERON_DKPI, frame.GetY1(), Q_VALUE_CDEUTERON_DKPI, frame.GetY2())
        cd_line.SetLineColor(ROOT.kGray + 2)
        cd_line.SetLineStyle(ROOT.kDashed)
        cd_line.Draw("same")

        cd_text = ROOT.TLatex(Q_VALUE_CDEUTERON_DKPI + 0.002, c.GetUymax() - 0.5, "cd Q-value")
        cd_text.SetTextFont(42)
        cd_text.SetTextSize(0.035)
        cd_text.SetTextColor(ROOT.kGray + 2)
        cd_text.Draw("same")

        leg.Draw()

        c.SaveAs(f"{BASEPATH}/final_plots/calib_mass/calib_{plot_name}_{ptmin}_{ptmax}.pdf")

        cd_shift = ROOT.TGraphErrors(3)
        for i, (year, val) in enumerate(zip(YEARS, cd_extrap)):
            cd_shift.SetPoint(i, int(year), val[0])
            cd_shift.SetPointError(i, 0, val[1])
        cd_shift.SetTitle(";Year;Mass shift (MeV/#it{c}^{2})")
        cd_shift.SetName(f"cd_shift_vs_year_{cent_class}_{ptmin}_{ptmax}")
        cd_shift.Write()

        lc_shift = ROOT.TGraphErrors(3)
        for i, (year, val) in enumerate(zip(YEARS, lc_extrap)):
            lc_shift.SetPoint(i, int(year), val[0])
            lc_shift.SetPointError(i, 0, val[1])
        lc_shift.SetTitle(";Year;Mass shift (MeV/#it{c}^{2})")
        lc_shift.SetName(f"lc_shift_vs_year_{cent_class}_{ptmin}_{ptmax}")
        lc_shift.Write()

        for y in YEARS:
            if y not in cd_extrap_vs_pt:
                cd_extrap_vs_pt[y] = []
                lc_extrap_vs_pt[y] = []
            cd_extrap_vs_pt[y].append(cd_extrap[YEARS.index(y)])
            lc_extrap_vs_pt[y].append(lc_extrap[YEARS.index(y)])

    for y in YEARS:
        n_pt = len(pt_bins) - 1
        pt_x = array("d", pt_bins[:-1])
        pt_ex = array("d", [0.0] * n_pt)

        cd_vals = array("d", [v[0] for v in cd_extrap_vs_pt[y]])
        cd_errs = array("d", [v[1] for v in cd_extrap_vs_pt[y]])
        cd_extrap_graph = ROOT.TGraphErrors(n_pt, pt_x, cd_vals, pt_ex, cd_errs)
        cd_extrap_graph.SetTitle(f"cd_shift_vs_pt_{cent_class}_{y};#it{{p}}_{{T}} (GeV/#it{{c}});Mass shift (MeV/#it{{c}}^{{2}})")
        cd_extrap_graph.SetName(f"cd_shift_vs_pt_{cent_class}_{y}")
        cd_extrap_graph.Write()

        lc_vals = array("d", [v[0] for v in lc_extrap_vs_pt[y]])
        lc_errs = array("d", [v[1] for v in lc_extrap_vs_pt[y]])
        lc_extrap_graph = ROOT.TGraphErrors(n_pt, pt_x, lc_vals, pt_ex, lc_errs)
        lc_extrap_graph.SetTitle(f"lc_shift_vs_pt_{cent_class}_{y};#it{{p}}_{{T}} (GeV/#it{{c}});Mass shift (MeV/#it{{c}}^{{2}})")
        lc_extrap_graph.SetName(f"lc_shift_vs_pt_{cent_class}_{y}")
        lc_extrap_graph.Write()

out_file_vs_qvalue.Close()
