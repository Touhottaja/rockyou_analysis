import colorlog
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from pyspark.sql import SparkSession

LINE_READ_LIMIT = 1000000  # Limit the number of lines read from the file
HEATMAP_CHARACTER_COUNT = 8  # Number of characters to analyze in the heatmap

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


def fetch_passwords_using_regex(pandas_df: pd.DataFrame,
                                regex: str) -> pd.DataFrame:
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


def find_most_common_words(pandas_df: pd.DataFrame, limit: int) -> None:
    """
    Finds the most common words in the passwords and prints them.

    :param pandas_df: Pandas DataFrame containing a column of passwords.
    """
    logger.info(f"Finding the {limit} most common words in the passwords")

    # Get a list of all words in the passwords
    all_words = " ".join(pandas_df["value"]).split()

    # Count the frequency of each word
    word_counts = Counter(all_words)

    # Print the most common words
    for word, count in word_counts.most_common(limit):
        logger.info(f"{word}: {count} occurrences")

    # Check if the most common words are used as substrings, add the occurrences
    # of the substrings to the count
    for word, count in word_counts.most_common(limit):
        substring_count = 0
        for password in pandas_df["value"]:
            if word in password:
                substring_count += 1
        logger.info(f"{word} (substring): {substring_count} occurrences")


def plot_character_heatmap(pandas_df: pd.DataFrame,
                           character_count: int) -> None:
    """
    Creates a heatmap showing the frequency of each character at each position
    in the passwords.

    :param pandas_df: Pandas DataFrame containing a column of passwords.
    """
    logger.info("Creating character frequency heatmap")

    # Define the characters to analyze
    characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=.,<>?;:[]{}|\\/'
    # Initialize a matrix to track character frequencies by position
    heatmap_data = {char: [0]*character_count for char in characters}

    for password in pandas_df['value']:
        for position, char in enumerate(password[:character_count]):
            if char.lower() in characters:
                heatmap_data[char.lower()][position] += 1

    # Convert the dictionary to a DataFrame for easier plotting and transpose
    # for seaborn heatmap compatibility
    heatmap_df = pd.DataFrame(heatmap_data).T

    # Plot the heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"Character Frequency by Position in Passwords For the First " \
              f"{LINE_READ_LIMIT} Passwords")
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

    # Filter the DataFrame to only include passwords of length 16 or less
    pandas_df = pandas_df[pandas_df["pw_len"] <= 16]

    # Plot histogram of password lengths
    plt.figure(figsize=(10, 6))
    pandas_df["pw_len"].hist(bins=range(pandas_df["pw_len"].min(),
                                        pandas_df["pw_len"].max() + 2),
                             color="skyblue",
                             edgecolor="black")
    plt.title(f"Histogram of Password Lengths for the First " \
              f"{LINE_READ_LIMIT} Passwords")
    plt.xlabel("Password Length")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0)
    plt.grid(axis="x", alpha=0)

    # Set xticks to each integer value in the range of password lengths
    plt.xticks(range(pandas_df["pw_len"].min(), pandas_df["pw_len"].max() + 1))

    # Save the plot
    plt.savefig(f"{PLOTS_DIR}/password_length_histogram.png")
    logger.info(f"Saved histogram of password lengths to " \
                f"{PLOTS_DIR}/password_length_histogram.png")


def plot_word_length(pandas_df: pd.DataFrame) -> None:
    """
    creates a box plot of the lengths of the words in the passwords.

    :param pandas_df: Pandas DataFrame containing a column of passwords.
    """
    logger.info("Creating box plot of word lengths")

    # Measure the length of each word
    pandas_df["word_len"] = pandas_df["value"].apply(len)

    # Show the length range
    logger.info(f"Word length range: {pandas_df['word_len'].min()} to " \
                f"{pandas_df['word_len'].max()} characters")

    # Show the shortest and longest words
    logger.info("Shortest word: " \
                f"{pandas_df['value'][pandas_df['word_len'].idxmin()]}")
    logger.info("Longest word: " \
                f"{pandas_df['value'][pandas_df['word_len'].idxmax()]}")

    # Show the average word length
    logger.info(f"Average word length: {pandas_df['word_len'].mean()} " \
                "characters")

    # Show the median word length
    logger.info(f"Median word length: {pandas_df['word_len'].median()} " \
                "characters")

    # Show the standard deviation of word lengths
    logger.info(f"Standard deviation of word lengths: " \
                f"{pandas_df['word_len'].std()} characters")

    # Remove passwords with word lengths greater than 16
    pandas_df = pandas_df[pandas_df["word_len"] <= 16]

    # Plot the word length box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(pandas_df["word_len"])
    plt.title(f"Box Plot of Word Lengths For the First {LINE_READ_LIMIT} " \
              "Passwords (Words <= 16 Characters)")
    plt.ylabel("Password length")

    # Plot y-axis every 5 characters
    plt.yticks(range(0, pandas_df["word_len"].max() + 1, 1))

    # Set grid style
    plt.grid(axis="y", alpha=0.75)

    # Save the plot
    plt.savefig(f"{PLOTS_DIR}/word_length_boxplot.png")
    logger.info(f"Saved box plot of word lengths to " \
                f"{PLOTS_DIR}/word_length_boxplot.png")


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


def main() -> None:
    """
    Main execution function.
    """
    # Read and limit data, then convert to Pandas DataFrame
    df = read_and_limit_data(ROCKYOU_FILE, LINE_READ_LIMIT)
    pandas_df = df.toPandas()

    ###################
    ## Plot examples ##
    ###################

    # Plot password lengths histogram
    #plot_password_lengths(pandas_df)

    # Plot character heatpmap
    #plot_character_heatmap(pandas_df, HEATMAP_CHARACTER_COUNT)

    # Plot word length box plot
    #plot_word_length(pandas_df)

    # Find most common words
    #find_most_common_words(pandas_df, 20)

    #################################
    ## Regular expression examples ##
    #################################

    # Fetch the count of numeric passwords
    #df = fetch_passwords_using_regex(pandas_df, "^[0-9]+$")
    #logger.info(f"Numeric passwords: {df.count()}")

    # Fetch alphabetic passwords
    #df = fetch_passwords_using_regex(pandas_df, "^[A-Za-z]+$")
    #logger.info(f"Alphabetic passwords: {df.count()}")

    # Fetch passwords with special characters
    #df = fetch_passwords_using_regex(pandas_df, "[!@#$%^&*(),.?\":{}|<>]+")
    #logger.info(f"Passwords with special characters: {df.count()}")

    # Fetch passwords that contain alphanumeric characters and special characters
    #df = fetch_passwords_using_regex(pandas_df,
    #    "^(?=.*[A-Za-z0-9])(?=.*[!@#$%^&*(),.?\":{}|<>]).+$")
    #logger.info(f"Passwords with alphanumeric and special characters: {df.count()}")

    # Fetch passwords that are shorter than 5 characters
    #df = fetch_passwords_using_regex(pandas_df, "^.{1,5}$")
    #logger.info(f"Passwords shorter than 5 characters: {df.count()}")

    # Fetch passwords that are between 6-8 characters
    #df = fetch_passwords_using_regex(pandas_df, "^.{0,16}$")
    #logger.info(f"Passwords between 6-8 characters: {df.count()}")

    # Fetch passwords with common patterns (e.g. "abc" or "123")
    # Find patterns that contain abc, e.g. abc and abcdefg
    #df = fetch_passwords_using_regex(pandas_df, "(abc)")
    #logger.info(f"Passwords with 'abc' patterns: {df.count()}")

    #df = fetch_passwords_using_regex(pandas_df, "(123)")
    #logger.info(f"Passwords with '123' patterns: {df.count()}")

    #df = fetch_passwords_using_regex(pandas_df, "(012)")
    #logger.info(f"Passwords with '012' patterns: {df.count()}")

    #df = fetch_passwords_using_regex(pandas_df, "(qwe)")
    #logger.info(f"Passwords with 'qwer patterns: {df.count()}")

    #df = fetch_passwords_using_regex(pandas_df, "(asd)")
    #logger.info(f"Passwords with 'asd' patterns: {df.count()}")

    #df = fetch_passwords_using_regex(pandas_df,
    #    "^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*(),.?\":{}|<>]).{8,}$")
    #logger.info("Passwords with alphanumeric and special characters:" \
    #            f"{df.count()}")


if __name__ == "__main__":
    main()
