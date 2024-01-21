# class SyntheticDataset(MIMIC3Dataset):

#     @classmethod
#     def make_synthetic_admissions(cls, colname, n_subjects):
#         rng = random.Random(0)

#         d1 = datetime.strptime('1/1/1950 1:30 PM', '%m/%d/%Y %I:%M %p')
#         d2 = datetime.strptime('1/1/2050 4:50 AM', '%m/%d/%Y %I:%M %p')
#         adms = []
#         for subject_id in range(n_subjects):
#             adm_d1 = random_date(d1, d2, rng)

#             n_admissions = rng.randint(1, 10)
#             adm_dates = sorted([
#                 random_date(adm_d1, d2, rng) for _ in range(2 * n_admissions)
#             ])
#             admittime = adm_dates[:n_admissions:2]
#             dischtime = adm_dates[1:n_admissions:2]
#             adms.append((subject_id, admittime, dischtime))
#         df = pd.DataFrame(adms,
#                           columns=list(colname[k]
#                                        for k in ('subject_id', 'admittime',
#                                                  'dischtime')))
#         df.index.names = [colname.index]
#         return df

#     @classmethod
#     def make_synthetic_dx(cls, dx_colname, dx_source_scheme, n_dx_codes,
#                           adm_df, adm_colname):
#         rng = random.Random(0)
#         n_dx_codes = rng.choices(range(n_dx_codes), k=len(adm_df))
#         codes = []
#         for i, adm_id in enumerate(adm_df.index):
#             n_codes = n_dx_codes[i]
#             codes.extend(
#                 (adm_id, c)
#                 for c in rng.choices(dx_source_scheme.codes, k=n_codes))

#         return pd.DataFrame(codes, columns=[dx_colname.index, dx_colname.code])

#     @classmethod
#     def make_synthetic_demographic(cls,
#                                    demo_colname,
#                                    adm_df,
#                                    adm_colname,
#                                    ethnicity_scheme=None,
#                                    gender_scheme=None):
#         rng = random.Random(0)
#         subject_ids = adm_df[adm_colname.subject_id].unique()
#         demo = []
#         for subject_id, df in adm_df.groupby(adm_colname.subject_id):
#             first_adm = df[adm_colname.admittime].min()
#             if gender_scheme is None:
#                 gender = None
#             else:
#                 gender = rng.choices(gender_scheme.codes, k=1)[0]

#             if ethnicity_scheme is None:
#                 ethnicity = None
#             else:
#                 ethnicity = rng.choices(ethnicity_scheme.codes, k=1)[0]

#             date_of_birth = random_date(first_adm - relativedelta(years=100),
#                                         first_adm - relativedelta(years=18),
#                                         rng)

#             row = (subject_id, date_of_birth) + (c for c in (gender, ethnicity)
#                                                  if c is not None)

#         columns = [demo_colname.subject_id, demo_colname.date_of_birth]
#         if gender_scheme is not None:
#             columns.append(demo_colname.gender)
#         if ethnicity_scheme is not None:
#             columns.append(demo_colname.ethncity)

#         df = pd.DataFrame(demo, columns=columns)
#         return df
