import colorlog
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession

LINE_READ_LIMIT = 1000  # Limit the number of lines read from the file

# Files and directories
ROCKYOU_FILE = "rockyou.txt"
PLOTS_DIR = "plots"

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    })

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def read_and_limit_data(filepath, limit=1000) -> pd.DataFrame:
    """
    Reads data from a text file using Spark and limits the results.

    :param filepath: Path to the text file.
    :param limit: Number of rows to limit the DataFrame to.
    :return: Spark DataFrame limited to 'limit' rows.
    """
    logger.info(f"Reading data from {filepath} and limiting to {limit} rows")
    spark = SparkSession.builder \
        .master("local") \
        .appName("rockyou") \
        .getOrCreate()

    df = spark.read.text(filepath)
    limited_df = df.limit(limit)
    return limited_df


def plot_password_lengths(df) -> None:
    """
    Plots a histogram of password lengths.

    :param df: Spark DataFrame containing a column of passwords.
    """
    logger.info("Plotting histogram of password lengths")

    # Convert the Spark DataFrame to a Pandas DataFrame
    pandas_df = df.toPandas()

    # Calculate password lengths
    pandas_df["pw_len"] = pandas_df["value"].apply(len)

    # Plot histogram of password lengths
    plt.figure(figsize=(10, 6))
    pandas_df["pw_len"].hist(bins=30, color="skyblue", edgecolor="black")
    plt.title("Histogram of Password Lengths for the First 1000 Passwords")
    plt.xlabel("Password Length")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    # Save the plot
    plt.savefig(f"{PLOTS_DIR}/password_length_histogram.png")
    logger.info(f"Saved histogram of password lengths to " \
                 "{PLOTS_DIR}/password_length_histogram.png")


def main() -> None:
    """
    Main execution function.
    """
    df = read_and_limit_data(ROCKYOU_FILE, LINE_READ_LIMIT)
    plot_password_lengths(df)


if __name__ == "__main__":
    main()
