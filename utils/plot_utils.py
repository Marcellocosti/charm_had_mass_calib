import ROOT
from ROOT import gStyle, TCanvas, TLegend

MARKERS = {
    "2023": 20,
    "2024": 21,
    "2025": 22,
}

colors = {
    "2023": ROOT.kRed+1,
    "2024": ROOT.kBlue+1,
    "2025": ROOT.kGreen+2
}

# Global publication style
def set_figure_style():

    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    gStyle.SetFrameLineWidth(2)
    gStyle.SetHistLineWidth(2)
    gStyle.SetLineWidth(2)

    gStyle.SetLegendBorderSize(0)
    gStyle.SetEndErrorSize(4)

    gStyle.SetLabelFont(42, "XYZ")
    gStyle.SetTitleFont(42, "XYZ")

    gStyle.SetLabelSize(0.042, "XYZ")
    gStyle.SetTitleSize(0.048, "XYZ")

    gStyle.SetTitleOffset(1.15, "X")
    gStyle.SetTitleOffset(1.45, "Y")


# Canvas
def make_canvas(name):

    c = TCanvas(name, "", 700, 650)

    c.SetLeftMargin(0.14)
    c.SetRightMargin(0.04)
    c.SetBottomMargin(0.13)
    c.SetTopMargin(0.05)

    return c


# Legend
def make_legend(x1=0.18, y1=0.72, x2=0.40, y2=0.88):

    leg = TLegend(x1, y1, x2, y2)

    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.040)

    return leg


def set_graph_style(g, year):

    g.SetMarkerStyle(MARKERS[year])
    g.SetMarkerSize(1.35)

    g.SetMarkerColor(colors[year])
    g.SetLineColor(colors[year])
    g.SetLineWidth(2)


# Draw decorations
def set_plot_style(mg, xtitle, variable, labels, particle=None, cent_class=None, pt_range=None, title=None):

    mg.Draw("AP")

    x = mg.GetXaxis()
    y = mg.GetYaxis()

    x.SetTitle(xtitle)

    if particle is None:
        y.SetTitle(f"M_{{meas}} - M_{{PDG}} (MeV/#it{{c}}^{{2}})")
    else:
        ylabel = labels[variable][0]
        if callable(ylabel):
            y.SetTitle(ylabel(particle))
        else:
            y.SetTitle(ylabel)

    x.SetTitleSize(0.048)
    y.SetTitleSize(0.048)

    x.SetLabelSize(0.042)
    y.SetLabelSize(0.042)

    x.SetTitleOffset(1.15)
    y.SetTitleOffset(1.45)

    ROOT.gPad.Update()

    h = mg.GetHistogram()

    ymin = h.GetMinimum()
    ymax = h.GetMaximum()

    dy = ymax - ymin
    if dy == 0:
        dy = abs(ymax) if ymax else 1.

    mg.SetMinimum(ymin - 0.15 * dy)
    mg.SetMaximum(ymax + 0.15 * dy)

    objects = []

    # reference line
    ref = labels[variable][1]

    if ref is not None:

        yref = ref(particle) if particle else ref(None)

        line = ROOT.TLine(x.GetXmin(), yref, x.GetXmax(), yref)

        line.SetLineStyle(ROOT.kDashed)
        line.SetLineColor(ROOT.kGray + 2)
        line.SetLineWidth(2)

        line.Draw()

        objects.append(line)

    latex = ROOT.TLatex()
    ypos = 0.87
    if cent_class:
        latex.DrawLatex(0.18, ypos, f"Centrality {cent_class.replace('_','-')}%")
        ypos -= 0.05

    if pt_range:
        latex.DrawLatex(0.18, ypos, f"{pt_range[0]} < #it{{p}}_{{T}} < {pt_range[1]} GeV/#it{{c}}")

    objects.append(latex)

    return objects
