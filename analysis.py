# %%
# import libraries
import numpy as np
import pandas as pd
import scipy.stats as st

# %%
# load dataset
df = pd.read_csv("PI_Covid_Total.csv")

# %%
#  ######  ##       ########    ###    ##    ## #### ##    ##  ######
# ##    ## ##       ##         ## ##   ###   ##  ##  ###   ## ##    ##
# ##       ##       ##        ##   ##  ####  ##  ##  ####  ## ##
# ##       ##       ######   ##     ## ## ## ##  ##  ## ## ## ##   ####
# ##       ##       ##       ######### ##  ####  ##  ##  #### ##    ##
# ##    ## ##       ##       ##     ## ##   ###  ##  ##   ### ##    ##
#  ######  ######## ######## ##     ## ##    ## #### ##    ##  ######
# remove columns I am not using
raw_iat_columns = [
    "Practice_Healthy_ER",
    "Practice_Infected_ER",
    "Practice_Healthy_RT",
    "Practice_Infected_RT",
    "USA_Healthy_RT",
    "USA_Infected_RT",
    "USA_SD",
    "USA_DD_Score",
    "USA_Overall_ER",
    "US_Overall_Fast",
    "UK_Healthy_RT",
    "UK_Infected_RT",
    "UK_SD",
    "UK_DD_Score",
    "UK_Overall_ER",
    "UK_Overall_Fast",
    "Italy_Healthy_RT",
    "Italy_Infected_RT",
    "Italy_SD",
    "Italy_DD_Score",
    "Italy_Overall_ER",
    "Italy_Overall_Fast",
    "China_Healthy_RT",
    "China_Infected_RT",
    "China_SD",
    "China_DD_Score",
    "China_Overall_ER",
    "China_Overall_Fast",
    "China_Trials_N",
    "italy_Trials_N",
    "practice_Trials_N",
    "UK_Trial_N",
    "USA_Trials_N",
]  # not enough information to analyze

reversed_coded_original_columns = [
    "indiv2",
    "indiv5",
    "indiv6",
    "pvd3",
    "pvd5",
    "pvd11",
    "pvd12",
    "pvd13",
    "pvd14",
    "pvd17",
    "pvd19",
    "pvd21",
    "pvd23",
]  # the scrambling was done after reverse-coding, and thus the original
# does not correspond to the reversed-coded columns
# I chose to treat the reverse-coded columns as the actual responses

irrelevant_columns = [
    "session_id",
    "referrer",
    "broughtwebsite",
    "study_url",
    "user_agent",
]

inconsistent_columns = [
    "month",
    "day",
    "Week1_7",
    "year",
    "hour",
    "weekday",
    "sex",
    "gender01",
    "CountryresUS_Not",
    "religion_switch",
    "Race_number",
    "Race_White_Asian_Black_Other",
    "Race_White_Other",
    "PVD_Infectability",
    "PVD_Germs",
    "Germ_Ingroup",
    "Germ_outgroup",
    "Total_PVD",
    "Mean_Individualism",
    "Lib_Moderate_Conservative",
]  # these columns are inconsistent
# I chose only a subset of columns as the actual responses

remove_columns = (
    raw_iat_columns
    + reversed_coded_original_columns
    + irrelevant_columns
    + inconsistent_columns
)
df = df.drop(remove_columns, axis=1)

# %%
# replace space as missing data
df = df.replace(r"^\s*$", np.nan, regex=True)

# %%
# recalculate summary scores
# the scrambling made the missing data unnatural and impossible to analyze
# I averaged whatever values' left to analyze
df["individualism"] = (
    df[["indiv1", "indiv3", "indiv4", "indivR2", "indivR5", "indivR6"]]
    .astype(float)
    .mean(axis=1)
)

df["pvd_perc_infect"] = (
    df[
        [
            "pvd2",
            "rpvd5",
            "pvd6",
            "pvd8",
            "pvd10",
            "rpvd12",
            "rpvd14",
        ]
    ]
    .astype(float)
    .mean(axis=1, skipna=True)
)

df["pvd_germ_aver"] = (
    df[
        [
            "pvd1",
            "rpvd3",
            "pvd4",
            "pvd7",
            "pvd9",
            "rpvd11",
            "rpvd13",
            "pvd15",
        ]
    ]
    .astype(float)
    .mean(axis=1)
)

df["pvd_germ_in"] = (
    df[
        [
            "pvd16",
            "rpvd17",
            "pvd18",
            "rpvd19",
        ]
    ]
    .astype(float)
    .mean(axis=1)
)

df["pvd_germ_out"] = (
    df[
        [
            "pvd20",
            "rpvd21",
            "pvd22",
            "rpvd23",
        ]
    ]
    .astype(float)
    .mean(axis=1)
)

# %%
# remove individual items from summarized scales
item_columns = [
    "indiv1",
    "indiv3",
    "indiv4",
    "indivR2",
    "indivR5",
    "indivR6",
    "pvd1",
    "pvd2",
    "pvd4",
    "pvd6",
    "pvd7",
    "pvd8",
    "pvd9",
    "pvd10",
    "pvd15",
    "pvd16",
    "pvd18",
    "pvd20",
    "pvd22",
    "rpvd3",
    "rpvd5",
    "rpvd11",
    "rpvd12",
    "rpvd13",
    "rpvd14",
    "rpvd17",
    "rpvd19",
    "rpvd21",
    "rpvd23",
]  # these columns are inconsistent
# I chose only a subset of columns as the actual responses

df = df.drop(item_columns, axis=1)

# %%
# reorder columns
df = df[
    [
        "date",
        "birth_sex",
        "gender",
        "countrycit_num",
        "countryres_num",
        "edu",
        "fieldofstudy",
        "religiosity",
        "ses",
        "ethnicityomb",
        "raceombmulti",
        "Age",
        "econpoliticalid",
        "socialpoliticalid",
        "politicalid",
        "TAmericans_0to10",
        "Tasian_0to10",
        "Tblack_0to10",
        "TBritons_0to10",
        "Tbrown_0to10",
        "TChinese_0to10",
        "TClose_0to10",
        "TCough_0to10",
        "TIsolate_0to10",
        "TItalians_0to10",
        "TWash_0to10",
        "Twhite_0to10",
        "attexperts",
        "TScientists_0to10",
        "TMedical_0to10",
        "TGoverment_0to10",
        "att9Americans",
        "att9Britons",
        "att9Italians",
        "att9Chinese",
        "attCoronaWorried",
        "USA_IAT_Score_Cleaned",
        "UK_IAT_Score_Cleaned",
        "Italy_IAT_Score_Cleaned",
        "China_IAT_Score_Cleaned",
        "prioriat",
        "individualism",
        "pvd_perc_infect",
        "pvd_germ_aver",
        "pvd_germ_in",
        "pvd_germ_out",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
    ]
]

# %%
#    ###    ##    ##    ###    ##       ##    ##  ######  ####  ######
#   ## ##   ###   ##   ## ##   ##        ##  ##  ##    ##  ##  ##    ##
#  ##   ##  ####  ##  ##   ##  ##         ####   ##        ##  ##
# ##     ## ## ## ## ##     ## ##          ##     ######   ##   ######
# ######### ##  #### ######### ##          ##          ##  ##        ##
# ##     ## ##   ### ##     ## ##          ##    ##    ##  ##  ##    ##
# ##     ## ##    ## ##     ## ########    ##     ######  ####  ######
# replicate known phenomenon: in-group bias
in_out = df[["pvd_germ_in", "pvd_germ_out"]].dropna(axis=0)
in_mean, out_mean = in_out.mean()
t, p = st.ttest_rel(in_out["pvd_germ_in"], in_out["pvd_germ_out"])
# out-group germ aversion is stronger than in-group germ aversion


# %%
# replicate known phenomenon: conservatism correlates with disease avoidance
def r_to_p(r):
    t = abs(r) * ((n - 2) ** 0.5) / ((1 - r**2) ** 0.5)
    return 2 * (1 - st.t.cdf(t, df=n - 2))


politics = (
    df[
        [
            "econpoliticalid",
            "socialpoliticalid",
            "politicalid",
            "pvd_perc_infect",
            "pvd_germ_aver",
        ]
    ]
    .dropna(axis=0)
    .astype(float)
)
n = len(politics)
rs = politics.corr().loc[["pvd_perc_infect", "pvd_germ_aver"]][
    ["econpoliticalid", "socialpoliticalid", "politicalid"]
]
ps = rs.apply(r_to_p)
# political liberalism weakly predicts disease avoidance

# %%
# being a foreigner lead to stronger disease avoidance?
country = df[["countrycit_num", "countryres_num"]].dropna(axis=0)
df["foreigner"] = country["countrycit_num"] != country["countryres_num"]

foreign = (
    df[
        [
            "foreigner",
            "pvd_perc_infect",
            "pvd_germ_aver",
        ]
    ]
    .dropna(axis=0)
    .astype(float)
)
foreigner = foreign[foreign["foreigner"] == True]
local = foreign[foreign["foreigner"] == False]

t, p = st.ttest_ind(foreigner["pvd_perc_infect"], local["pvd_perc_infect"])
# no effect
t, p = st.ttest_ind(foreigner["pvd_germ_aver"], local["pvd_germ_aver"])
# foreigners have stronger germ aversion compare to locals

# %%
# does night time lead to more fear for diseases?
df["time"] = df["date"].str.split(" ").str[1]
df["hour"] = df["time"].str.split(":").str[0]
# assuming this is the local time for the subjects
df["night"] = (df["hour"].astype("int") >= 19) | (df["hour"].astype("int") <= 3)
# 7PM - 3AM

nighttime = df[
    [
        "night",
        "pvd_perc_infect",
        "pvd_germ_aver",
    ]
].dropna(axis=0)
night = nighttime[nighttime["night"] == True]
day = nighttime[nighttime["night"] == False]

t, p = st.ttest_ind(night["pvd_perc_infect"], day["pvd_perc_infect"])
# no effect
t, p = st.ttest_ind(night["pvd_germ_aver"], day["pvd_germ_aver"])
# no effect


# %%
# I ran out of time to explore the dataset
# What I'd do next would be exploratory analyses, including
#   1. overall pairwise correlations
#   2. predict pvd or vaccine questionnaire with linear regressions
#   3. running the above analyses controlling for gender, race, political view
#   4. Perhaps plot some scatter plots and distributions to understand the data
