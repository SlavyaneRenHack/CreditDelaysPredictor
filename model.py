import pandas as pd
import joblib


base_df_columns = [
	'CLIENT_ID',
	'AGE',
	'MAN',
	'INCOME',
	'IP_FLAG',
	'SME_FLAG',
	'REFUGEE_FLAG',
	'PDN',
	'paid_off_count',
	'active_count',
	'active_sum',
	'max_term',
	'credits_delay_count',
	'cards_count',
	'cards_delay_count',
	'MATRIAL_Вдовец / Вдова',
	'MATRIAL_Гражданский брак',
	'MATRIAL_Женат / замужем',
	'MATRIAL_Не женат / не замужем',
	'MATRIAL_Неизвестно',
	'MATRIAL_Разведен / Разведена'
]


card_df_columns = [
	'CLIENT_ID',
	'AGE',
	'MAN',
	'INCOME',
	'IP_FLAG',
	'SME_FLAG',
	'REFUGEE_FLAG',
	'PDN',
	'paid_off_count',
	'active_count',
	'active_sum',
	'max_term',
	'credits_delay_count',
	'cards_count',
	'MATRIAL_Вдовец / Вдова',
	'MATRIAL_Гражданский брак',
	'MATRIAL_Женат / замужем',
	'MATRIAL_Не женат / не замужем',
	'MATRIAL_Неизвестно',
	'MATRIAL_Разведен / Разведена',
	'CС_LIMIT_NVAL',
	'СС_GRACE_PERIOD',
	'CURR_RATE',
	# 'delay',
	'PC_bb',
	'PC_blockbuster',
	'PC_grace',
	'PC_other'
]

credit_df_columns = [
	'CLIENT_ID',
	'AGE',
	'MAN',
	'INCOME',
	'IP_FLAG',
	'SME_FLAG',
	'REFUGEE_FLAG',
	'PDN',
	'paid_off_count',
	'active_count',
	'active_sum',
	'max_term',
	'credits_delay_count',
	'cards_count',
	'cards_delay_count',
	'MATRIAL_Вдовец / Вдова',
	'MATRIAL_Гражданский брак',
	'MATRIAL_Женат / замужем',
	'MATRIAL_Не женат / не замужем',
	'MATRIAL_Неизвестно',
	'MATRIAL_Разведен / Разведена',
	'TERM',
	'ORIG_AMOUNT',
	'CURR_RATE_NVAL',
	# 'delay',
	'CT_501',
	'CT_503',
	'CT_506',
	'CT_704',
	'CT_705',
	'CT_707',
	'CT_708',
	'CT_709',
	'PC_GP Cross-Sale',
	'PC_GP Direct',
	'PC_GP External Refinance',
	'PC_GP Refinance',
	'PC_GP Top-Up',
	'PC_Promo Beneficial Installment Loan',
	'PC_Promo Rate Installment Loan',
	'PC_Standard Installment Loan'
]


def prepare_base_df(user_data, credit_df, card_df):
	# User data
	df = pd.DataFrame([user_data])
	df = df.drop(columns=["JOB", "EMPLOYEE_FLAG"])
	df = df.rename(columns={"GENDER": "MAN"})

	df["MAN"] = (df["MAN"] - 1).astype("bool")

	# Credit data
	credit_df["TERM"] = credit_df["TERM"].map(lambda x: x[:-1]).astype("int64")

	current_date = pd.Timestamp.now()
	def mounth_until_end(open_date, term):
		mounth_passed = (current_date.year - open_date.year) * 12 + current_date.month - open_date.month
		reamin_mounths = term - mounth_passed
		return max(0, reamin_mounths)

	credit_df["REAMIN_MOUNTHS"] = credit_df.apply(lambda row: mounth_until_end(row['VALUE_DT'], row['TERM']), axis=1)
	credit_df = credit_df.rename(columns={"OVERDUE_IND": "delay"})

	# Cards data
	card_df = card_df.drop(columns=["CARD_TYPE", "OPEN_DT"])
	card_df = card_df.rename(columns={"СС_OVERDUE_IND": "delay"})

	def replace_product_code(value):
		value = value.lower()
		if 'grace' in value:
			return 'grace'
		elif 'blockbuster' in value:
			return 'blockbuster'
		elif '_bb_' in value:
			return 'bb'

		return 'other'

	card_df["PRODUCT_CODE"] = card_df["PRODUCT_CODE"].apply(replace_product_code)

	# Merging
	## Preparing dataframes
	df_paid_off_counts = credit_df[credit_df["REAMIN_MOUNTHS"] == 0].groupby("CLIENT_ID").size().reset_index(name="paid_off_count")
	df_active_counts = credit_df[credit_df["REAMIN_MOUNTHS"] > 0].groupby("CLIENT_ID").size().reset_index(name="active_count")
	df_active_sums = credit_df[credit_df["REAMIN_MOUNTHS"] > 0].groupby("CLIENT_ID")["ORIG_AMOUNT"].sum().reset_index(name="active_sum")
	df_max_term = credit_df.groupby('CLIENT_ID')['TERM'].max().reset_index(name='max_term')
	df_delay_counts = credit_df[credit_df['delay'] > 0].groupby("CLIENT_ID").size().reset_index(name="credits_delay_count")

	df_cards_count = card_df.groupby('CLIENT_ID').size().reset_index(name="cards_count")
	df_cards_delay_count = card_df[card_df['delay'] > 0].groupby('CLIENT_ID').size().reset_index(name="cards_delay_count")

	## Merge dataframes
	df = pd.merge(df, df_paid_off_counts, on='CLIENT_ID', how='left')
	df = pd.merge(df, df_active_counts, on='CLIENT_ID', how='left')
	df = pd.merge(df, df_active_sums, on='CLIENT_ID', how='left')
	df = pd.merge(df, df_max_term, on='CLIENT_ID', how='left')
	df = pd.merge(df, df_delay_counts, on='CLIENT_ID', how='left')

	df = pd.merge(df, df_cards_count, on='CLIENT_ID', how='left')
	df = pd.merge(df, df_cards_delay_count, on='CLIENT_ID', how='left')

	## Filling empty data
	df['paid_off_count'] = df['paid_off_count'].fillna(0).astype('int64')
	df['active_count'] = df['active_count'].fillna(0).astype('int64')
	df['max_term'] = df['max_term'].fillna(0).astype('int64')
	df['active_sum'] = df['active_sum'].fillna(0)
	df['credits_delay_count'] = df['credits_delay_count'].fillna(0).astype('int64')

	df['cards_count'] = df['cards_count'].fillna(0).astype('int64')
	df['cards_delay_count'] = df['cards_delay_count'].fillna(0).astype('int64')

	# Process object columns
	df = pd.get_dummies(df, columns=['MARITAL_STATUS'], prefix='MATRIAL')
	df = df.drop(columns=['REGION', 'ORGANIZATION'], axis=1)
	df = df.replace({False: 0, True: 1})

	for column in base_df_columns:
		if column not in df.columns:
			df[column] = 0

	return df[base_df_columns]


def prepare_card_df(base_df, card_df):
	tmp_df = card_df.copy()
	def replace_product_code(value):
		value = value.lower()
		if 'grace' in value:
			return 'grace'
		elif 'blockbuster' in value:
			return 'blockbuster'
		elif '_bb_' in value:
			return 'bb'

		return 'other'

	tmp_df["PRODUCT_CODE"] = tmp_df["PRODUCT_CODE"].apply(replace_product_code)
	tmp_df = pd.get_dummies(tmp_df, columns=['PRODUCT_CODE'], prefix='PC')

	df = pd.merge(base_df, tmp_df, on='CLIENT_ID', how='left')
	df = df.fillna(0)
	df = df.astype({col: 'int64' for col in df.select_dtypes(include='bool').columns})
	df = df.drop(columns=["cards_delay_count", "CARD_TYPE", "OPEN_DT", "СС_OVERDUE_IND"])

	for column in card_df_columns:
		if column not in df.columns:
			df[column] = 0

	return df[card_df_columns]


def prepare_credit_df(base_df, credit_df, credit_data):
	tmp_df = credit_df.copy()
	for key, value in credit_data.items():
		tmp_df[key] = value

	tmp_df = tmp_df.drop(columns=['CREDIT_PURCHASE', 'VALUE_DT', 'OVERDUE_IND', 'REAMIN_MOUNTHS'])
	tmp_df = pd.get_dummies(tmp_df, columns=['CREDIT_TYPE'], prefix='CT')
	tmp_df = pd.get_dummies(tmp_df, columns=['PRODUCT_CODE'], prefix='PC')

	df = pd.merge(base_df, tmp_df, on='CLIENT_ID', how='left')
	df = df.fillna(0)

	df = df.astype({col: 'int64' for col in df.select_dtypes(include='bool').columns})
	df["credits_delay_count"] = df["credits_delay_count"].apply(lambda x: max(x - 1, 0))

	for column in credit_df_columns:
		if column not in df.columns:
			df[column] = 0

	return df[credit_df_columns]


def main():
	credit_df = pd.read_excel("data/raw/credit_ds.xlsx")
	card_df = pd.read_excel("data/raw/card_ds.xlsx")
	user_data = {
		"CLIENT_ID": 1,
		"AGE": 24,
		"REGION": "Москва",
		"GENDER": 2,
		"JOB": None,
		"ORGANIZATION": "ЦИРК",
		"INCOME": 68000,
		"MARITAL_STATUS": "Неизвестно",
		"IP_FLAG": False,
		"SME_FLAG": False,
		"EMPLOYEE_FLAG": False,
		"REFUGEE_FLAG": False,
		"PDN": 26.6,
	}

	# credit_data = {
	# 	# "CLIENT_ID": 9039,
	# 	"CREDIT_TYPE": 501,
	# 	"CREDIT_PURCHASE": "Автокредит",
	# 	"PRODUCT_CODE": "Standard Installment Loan",
	# 	"TERM": 36,
	# 	"ORIG_AMOUNT": 69831,
	# 	"CURR_RATE_NVAL": 8.49,
	# 	"VALUE_DT": pd.to_datetime("2020-02-05"),
	# }

	model = joblib.load("params/dt_model.joblib")

	base_df = prepare_base_df(user_data, credit_df, card_df)
	# card_df = prepare_card_df(base_df, card_df)
	# credit_df = prepare_credit_df(base_df, credit_df, credit_data)

	print(base_df.info())
	# print(card_df.info())
	# print(credit_df.info())

	# card_df = card_df.drop(columns=['CLIENT_ID'])
	# credit_df = credit_df.drop(columns=['CLIENT_ID'])

	base_df = base_df.drop(columns=["CLIENT_ID"])
	pred = model.predict_proba(base_df)[0][0] * 100
	print("=" * 16)
	print(f"Client will repay the loan with probability: {pred:.2f}%")


if __name__ == '__main__':
	main()
