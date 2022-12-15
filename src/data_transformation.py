from data_preprocessor import *

raw_data = pd.read_csv("C:/Users/knola/Desktop/Final Year Project/Datasets/Raw/all_4G_data.csv", index_col=None)

raw_data = raw_data #[raw_data["movement_type"] == "static"]

imputed_data = WirelessDataPreProcessor(raw_data).get_df()

imputed_data.to_csv("transformed_4G_data.csv", index=False, encoding="utf-8")