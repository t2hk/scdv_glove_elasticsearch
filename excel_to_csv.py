import pandas as pd
import glob, sys

args = sys.argv
excel_dir = args[1]
output_csv_dir = args[2]

files = glob.glob(excel_dir + '/*.xls*')

for file in files:
  #excel = pd.ExcelFile(file, encoding='utf8')
  excel = pd.ExcelFile(file)

  print(file)

  sheet_names = excel.sheet_names
  for i, name in enumerate(sheet_names):
    if i > 0:
        continue

    csv_file = file.replace(excel_dir, output_csv_dir).replace(".xlsx", "").replace(".xls", "") + '_' + str(i) + '.csv'
    sheet_df = excel.parse(name, header=[0, 1])
    columns_val = sheet_df.columns.values

    col_names = []
    for col_vals in columns_val:
        # セル結合されているタイトル行は_で文字列結合する。
        # タイトルの分類名の括弧が全半角混在のため、半角に統一する。
        col_name = col_vals[0].replace('\n', '') + '_' + col_vals[1].replace('\n','')
        col_name = col_name.replace('（','(').replace('）',')')

        if 'Unnamed' in col_vals[1]:
            col_name = col_vals[0].replace('\n','')
        col_names.append(col_name)
    sheet_df.columns = col_names

    situation_col_name = '災害状況'
    if 'kikaisaigai' in file:
        situation_col_name = '災害発生状況'

    sheet_df[situation_col_name] = sheet_df[situation_col_name].replace('\r\n','', regex=True).replace('\r','', regex=True).replace('\n','', regex=True)

    sheet_df.to_csv(csv_file)
