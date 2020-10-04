import pandas as pd

# Move some of the reshaping-of-the-data out of the FullMixtureModel file
def curate_data(data_location, genotype,sheet=None):
	## Get the data to look nice
	df = pd.read_excel(io=data_location,sheetname=sheet)[['Genotype', 'Condition', 'Duration']]
	geno_durs = df.groupby(['Genotype']).get_group(genotype)
	grouped_durs = geno_durs.groupby('Condition')
	no_light = grouped_durs.get_group('No light')['Duration'].as_matrix()
	at_start = grouped_durs.get_group('At start')['Duration'].as_matrix()
	print pd.Categorical(geno_durs["Condition"]).categories
	return no_light, at_start, geno_durs

if __name__ == '__main__':
	curate_data("./Crz_ACR_Timing.xlsx",genotype="Crz>ACR")