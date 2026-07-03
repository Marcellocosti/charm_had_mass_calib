# Dplus

TOTAL_BRS_MC_CFG = 0.0752 + 0.0104 + 0.0156 + 0.0752

# 411:oneChannel = 1 0.0752 0 -321 211 211  # K- pi+ pi+
BR_PYTHIA_DPLUS_PIKPI = 0.0752 / TOTAL_BRS_MC_CFG
print(f"D+ ---> K- pi+ pi+ non-resonant: {BR_PYTHIA_DPLUS_PIKPI}")

# 411:addChannel = 1 0.0104 0 -313 211      # K*0 pi --> K- pi+ pi+
BR_PYTHIA_DPLUS_K0892PI_PIKPI = 0.0104 / TOTAL_BRS_MC_CFG
print(f"D+ ---> K*0 pi ---> K- pi+ pi+ : {BR_PYTHIA_DPLUS_K0892PI_PIKPI}")

# 411:addChannel = 1 0.0156 0 311 211     # K0 pi
BR_PYTHIA_DPLUS_K0SPI_PIPIPI = (0.0156 * 0.5) / TOTAL_BRS_MC_CFG     #  K0 decays to K0S with 50% probability
print(f"D+ ---> KK0 pi ---> K- pi+ pi+ : {BR_PYTHIA_DPLUS_K0SPI_PIPIPI}")

# 411:addChannel = 1 0.0752 0 333 211     # Phi Pi --> KKPi  # to have the same amount of D+->KKpi and D+->Kpipi
BR_PYTHIA_DPLUS_PHIPI_KKPI = (0.0752) / TOTAL_BRS_MC_CFG     #  K0 decays to K0S with 50% probability
print(f"D+ ---> KK0 pi ---> K- pi+ pi+ : {BR_PYTHIA_DPLUS_PHIPI_KKPI}")

print(f"\nSUMMARY:")
print(f"D+ ---> K- pi+ pi+ : {BR_PYTHIA_DPLUS_PIKPI + BR_PYTHIA_DPLUS_K0892PI_PIKPI}")
print(f"D+ ---> K- K+ pi+ : {BR_PYTHIA_DPLUS_PHIPI_KKPI}\n\n")

# Now Ds
TOTAL_BRS_MC_CFG_DS = 0.0400000 + 0.0440000

# 431:onIfMatch = 321 313    D_s -> K K*
BR_PYTHIA_DS_KKSTAR = 0.0400000 / TOTAL_BRS_MC_CFG_DS
print(f"D_s ---> K K* ---> K+ K- Pi+ : {BR_PYTHIA_DS_KKSTAR}")

# 431:onIfMatch = 333 211    D_s -> Phi pi
BR_PYTHIA_DS_PHIPI = 0.0440000 / TOTAL_BRS_MC_CFG_DS
print(f"D_s ---> Phi pi ---> K+ K- Pi+ : {BR_PYTHIA_DS_PHIPI}")

print(f"\nSUMMARY:")
print(f"----> D_s ---> K+ K- pi+ : {BR_PYTHIA_DS_KKSTAR + BR_PYTHIA_DS_PHIPI}")