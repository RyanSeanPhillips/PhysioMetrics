import pandas as pd
import sys

old_file = r'C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\Pleth_App_analysis\consolidated_data.xlsx'
new_file = r'C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\Pleth_App_analysis\consolidated_data_newformat2.xlsx'

try:
    old_xl = pd.ExcelFile(old_file, engine='openpyxl')
    print("OLD FORMAT SHEETS:")
    for sheet in old_xl.sheet_names:
        print(f"  {sheet}")

    print("\nNEW FORMAT SHEETS:")
    new_xl = pd.ExcelFile(new_file, engine='openpyxl')
    for sheet in new_xl.sheet_names:
        print(f"  {sheet}")

    # Read first few rows of each sheet to compare structure
    print("\n" + "="*80)
    print("COMPARING SHEET STRUCTURES:")
    print("="*80)

    for sheet in old_xl.sheet_names:
        print(f"\n\nOLD FORMAT - Sheet: {sheet}")
        print("-" * 80)
        try:
            df_old = pd.read_excel(old_file, sheet_name=sheet, nrows=10)
            print(f"Shape: {df_old.shape}")
            print(f"Columns: {list(df_old.columns)}")
            print("\nFirst few rows:")
            print(df_old.head())
        except Exception as e:
            print(f"Error reading: {e}")

    for sheet in new_xl.sheet_names:
        print(f"\n\nNEW FORMAT - Sheet: {sheet}")
        print("-" * 80)
        try:
            df_new = pd.read_excel(new_file, sheet_name=sheet, nrows=10)
            print(f"Shape: {df_new.shape}")
            print(f"Columns: {list(df_new.columns)}")
            print("\nFirst few rows:")
            print(df_new.head())
        except Exception as e:
            print(f"Error reading: {e}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
