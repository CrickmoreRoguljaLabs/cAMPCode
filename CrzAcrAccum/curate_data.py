import pandas as pd

# Move some of the reshaping-of-the-data out of the FullMixtureModel file
def curate_data(data_location, genotype):
	## Get the data to look nice
	df_dict = pd.read_excel(data_location, sheetname=None)
	off_at_10_data = df_dict["light off at 10 min"]["mating duration (min)"] - 11.0
	return off_at_10_data, df_dict

if __name__ == '__main__':
	curate_data("./Crz_ACR_Timing.xlsx",genotype="Crz>ACR")