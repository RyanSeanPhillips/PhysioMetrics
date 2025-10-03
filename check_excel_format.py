"""
Check Excel format differences between old and new consolidated files.
Since direct reading might fail if files are open, we'll try pandas with xlrd/openpyxl.
"""
import pandas as pd
import sys

old_file = r'C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\Pleth_App_analysis\consolidated_data.xlsx'
new_file = r'C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\Pleth_App_analysis\consolidated_data_newformat2.xlsx'

print("="*80)
print("ATTEMPTING TO READ EXCEL FILES")
print("="*80)

# Try reading new file (generated from NPZ)
try:
    print("\nReading NEW format file...")
    new_xl = pd.ExcelFile(new_file, engine='openpyxl')
    print(f"[OK] NEW file sheets: {new_xl.sheet_names}\n")

    # Show structure of first sheet
    if new_xl.sheet_names:
        first_sheet = new_xl.sheet_names[0]
        df = pd.read_excel(new_file, sheet_name=first_sheet, nrows=5)
        print(f"NEW - First sheet '{first_sheet}' structure:")
        print(f"  Columns ({len(df.columns)} total):")
        for i, col in enumerate(df.columns[:20]):  # First 20 columns
            print(f"    {i+1:3d}. {col}")
        if len(df.columns) > 20:
            print(f"    ... and {len(df.columns) - 20} more columns")
        print(f"\n  First row values (first 10 cols):")
        print(df.iloc[0, :10].to_dict())

except Exception as e:
    print(f"[ERROR] Error reading new file: {e}")
    import traceback
    traceback.print_exc()

# Try reading old file (generated from CSV)
try:
    print("\n" + "="*80)
    print("Reading OLD format file...")
    old_xl = pd.ExcelFile(old_file, engine='openpyxl')
    print(f"[OK] OLD file sheets: {old_xl.sheet_names}\n")

    # Show structure of first sheet
    if old_xl.sheet_names:
        first_sheet = old_xl.sheet_names[0]
        df = pd.read_excel(old_file, sheet_name=first_sheet, nrows=5)
        print(f"OLD - First sheet '{first_sheet}' structure:")
        print(f"  Columns ({len(df.columns)} total):")
        for i, col in enumerate(df.columns[:20]):  # First 20 columns
            print(f"    {i+1:3d}. {col}")
        if len(df.columns) > 20:
            print(f"    ... and {len(df.columns) - 20} more columns")
        print(f"\n  First row values (first 10 cols):")
        print(df.iloc[0, :10].to_dict())

except Exception as e:
    print(f"[ERROR] Error reading old file: {e}")
    print("\nThis is likely because:")
    print("  1. File is open in Excel (close it and try again)")
    print("  2. File is corrupted")
    print("  3. File is not a valid Excel file\n")
