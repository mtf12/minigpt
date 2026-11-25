# Run the whole data preparation from the terminal, in the same order as the notebook:
#   python models/preprocessing/run_preprocessing.py
#   python models/preprocessing/run_preprocessing.py --skip-reports   (cleaning, balancing and splitting only)
import argparse

import balance_languages
import clean_dataset
import data_exploration
import quality_report
import split_data


# Run every stage in order; the reports can be skipped because they are slow and only print.
def main():
    parser = argparse.ArgumentParser(description="Prepare the Arabic + English dataset.")
    parser.add_argument("--skip-reports", action="store_true", help="do not print the exploration and quality reports")
    args = parser.parse_args()

    if not args.skip_reports:
        data_exploration.explore("raw")
    light_stats = clean_dataset.run_light_cleaning()
    clean_stats = clean_dataset.run_heavy_cleaning(light_stats)
    if not args.skip_reports:
        quality_report.run(clean_stats)
    balance_languages.run()
    split_data.run()


if __name__ == "__main__":
    main()
