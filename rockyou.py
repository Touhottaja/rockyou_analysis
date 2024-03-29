import colorlog
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def fetch_passwords_using_regex(pandas_df: pd.DataFrame, regex: str) -> pd.DataFrame:
    """
    Fetches passwords from a DataFrame using a regular expression.

    :param pandas_df: Pandas DataFrame containing a column of passwords.
    :param regex: Regular expression to use for filtering.
    :return: DataFrame containing passwords that match the regular expression.
    """
    logger.info(f"Fetching passwords using regular expression: {regex}")

    # Filter passwords using regular expression
    filtered_df = pandas_df[pandas_df["value"].str.contains(regex)]
    return filtered_df


def plot_character_heatmap(pandas_df: pd.DataFrame) -> None:
    """
    Creates a heatmap showing the frequency of each character at each position in the passwords.

    :param pandas_df: Pandas DataFrame containing a column of passwords.
    """
    logger.info("Creating character frequency heatmap")

    # Define the characters to analyze
    characters = 'abcdefghijklmnopqrstuvwxyz0123456789'
    # Initialize a matrix to track character frequencies by position
    heatmap_data = {char: [0]*10 for char in characters}  # Adjust 10 to your desired password length analysis

    for password in df['value']:
        for position, char in enumerate(password[:10]):  # Adjust slice as needed
            if char.lower() in characters:
                heatmap_data[char.lower()][position] += 1

    # Convert the dictionary to a DataFrame for easier plotting
    heatmap_df = pd.DataFrame(heatmap_data).T  # Transpose for seaborn heatmap compatibility

    # Plot the heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Character Frequency by Position in Passwords")
    plt.xlabel("Position in Password")
    plt.ylabel("Character")
    plt.yticks(rotation=0)  # Keep character labels upright

    # Save the heatmap
    heatmap_filename = f"{PLOTS_DIR}/character_frequency_heatmap.png"
    plt.savefig(heatmap_filename)
    logger.info(f"Saved character frequency heatmap to {heatmap_filename}")

    plt.close()  # Close the plot to avoid displaying it inline if not desired


def plot_password_lengths(pandas_df: pd.DataFrame) -> None:
    """
    Plots a histogram of password lengths.

    :param pandas_df: Pandas DataFrame containing a column of passwords.
    """
    logger.info("Plotting histogram of password lengths")

    # Calculate password lengths
    pandas_df["pw_len"] = pandas_df["value"].apply(len)

    # Plot histogram of password lengths
    plt.figure(figsize=(10, 6))
    pandas_df["pw_len"].hist(bins=30, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of Password Lengths for the First " \
              f"{LINE_READ_LIMIT} Passwords")
    plt.xlabel("Password Length")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    # Save the plot
    plt.savefig(f"{PLOTS_DIR}/password_length_histogram.png")
    logger.info(f"Saved histogram of password lengths to " \
                f"{PLOTS_DIR}/password_length_histogram.png")


def read_and_limit_data(filepath: str, limit: int=1000) -> pd.DataFrame:
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


def plot_character_heatmap(df: pd.DataFrame) -> None:
    """
    Creates a heatmap showing the frequency of each character at each position in the passwords.

    :param pandas_df: Pandas DataFrame containing a column of passwords.
    """
    logger.info("Creating character frequency heatmap")

    # Define the characters to analyze
    characters = 'abcdefghijklmnopqrstuvwxyz0123456789'
    # Initialize a matrix to track character frequencies by position
    heatmap_data = {char: [0]*10 for char in characters}  # Adjust 10 to your desired password length analysis

    for password in df['value']:
        for position, char in enumerate(password[:10]):  # Adjust slice as needed
            if char.lower() in characters:
                heatmap_data[char.lower()][position] += 1

    # Convert the dictionary to a DataFrame for easier plotting
    heatmap_df = pd.DataFrame(heatmap_data).T  # Transpose for seaborn heatmap compatibility

    # Plot the heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Character Frequency by Position in Passwords")
    plt.xlabel("Position in Password")
    plt.ylabel("Character")
    plt.yticks(rotation=0)  # Keep character labels upright

    # Save the heatmap
    heatmap_filename = f"{PLOTS_DIR}/character_frequency_heatmap.png"
    plt.savefig(heatmap_filename)
    logger.info(f"Saved character frequency heatmap to {heatmap_filename}")

    plt.close()  # Close the plot to avoid displaying it inline if not desired


def main() -> None:
    """
    Main execution function.
    """
    # Read and limit data, then convert to Pandas DataFrame
    df = read_and_limit_data(ROCKYOU_FILE, LINE_READ_LIMIT)
    pandas_df = df.toPandas()

    # Plot password lengths histogram
    #plot_password_lengths(pandas_df)

    # Plot character heatpmap
    #plot_character_heatmap(pandas_df)

    ## Fetch numeric passwords
    #df = fetch_passwords_using_regex(pandas_df, "^[0-9]+$")
    #logger.info(f"Numeric passwords: {pandas_df}")
    ## Fetch alphabetic passwords
    #df = fetch_passwords_using_regex(pandas_df, "^[A-Za-z]+$")
    #logger.info(f"Alphabetic passwords: {pandas_df}")
    ## Fetch passwords with special characters
    #df = fetch_passwords_using_regex(pandas_df, "[!@#$%^&*(),.?\":{}|<>]+")
    #logger.info(f"Passwords with special characters: {pandas_df}")


if __name__ == "__main__":
    main()
