import colorlog
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, regexp_extract, lower, split, explode
from pyspark.sql.types import IntegerType

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


def fetch_passwords_using_regex(spark_df, regex: str):
    """
    Fetches passwords from a DataFrame using a regular expression.

    :param spark_df: Spark DataFrame containing a column of passwords.
    :param regex: Regular expression to use for filtering.
    :return: DataFrame containing passwords that match the regular expression.
    """
    logger.info(f"Fetching passwords using regular expression: {regex}")

    # Filter passwords using regular expression
    filtered_df = spark_df.filter(col("value").rlike(regex))
    return filtered_df


def find_most_common_words(spark_df, limit: int) -> None:
    """
    Finds the most common words in the passwords and prints them.

    :param spark_df: Spark DataFrame containing a column of passwords.
    """
    logger.info(f"Finding the {limit} most common words in the passwords")

    # Get a list of all words in the passwords
    words_df = spark_df.select(explode(split(col("value"), ' ')).alias("word"))

    # Count the frequency of each word
    word_counts_df = words_df.groupBy("word").count().orderBy(col("count").desc()).limit(limit)

    # Collect the results and print them
    for row in word_counts_df.collect():
        logger.info(f"{row['word']}: {row['count']} occurrences")

    # Check if the most common words are used as substrings, add the occurrences
    # of the substrings to the count
    for row in word_counts_df.collect():
        substring_count = spark_df.filter(col("value").contains(row['word'])).count()
        logger.info(f"{row['word']} (substring): {substring_count} occurrences")


def plot_character_heatmap(spark_df, character_count: int) -> None:
    """
    Creates a heatmap showing the frequency of each character at each position
    in the passwords.

    :param spark_df: Spark DataFrame containing a column of passwords.
    """
    logger.info("Creating character frequency heatmap")

    # Define the characters to analyze
    characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=.,<>?;:[]{}|\\/'

    # Initialize a matrix to track character frequencies by position
    heatmap_data = {char: [0]*character_count for char in characters}

    # Collect passwords and count character occurrences
    for row in spark_df.collect():
        password = row['value']
        for position, char in enumerate(password[:character_count]):
            if char in characters:
                heatmap_data[char][position] += 1

    # Convert the dictionary to a DataFrame for easier plotting and transpose
    heatmap_df = pd.DataFrame(heatmap_data).T

    # Plot the heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"Character Frequency by Position in Passwords For the First {LINE_READ_LIMIT} Passwords")
    plt.xlabel("Position in Password")
    plt.ylabel("Character")
    plt.yticks(rotation=0)  # Keep character labels upright

    # Save the heatmap
    heatmap_filename = f"{PLOTS_DIR}/character_frequency_heatmap.png"
    plt.savefig(heatmap_filename)
    logger.info(f"Saved character frequency heatmap to {heatmap_filename}")

    plt.close()  # Close the plot to avoid displaying it inline if not desired


def plot_password_lengths(spark_df) -> None:
    """
    Plots a histogram of password lengths.

    :param spark_df: Spark DataFrame containing a column of passwords.
    """
    logger.info("Plotting histogram of password lengths")

    # Calculate password lengths
    spark_df = spark_df.withColumn("pw_len", length(col("value")))

    # Filter the DataFrame to only include passwords of length 16 or less
    spark_df = spark_df.filter(col("pw_len") <= 16)

    # Collect data for plotting
    password_lengths = spark_df.select("pw_len").rdd.flatMap(lambda x: x).collect()

    # Plot histogram of password lengths
    plt.figure(figsize=(10, 6))
    plt.hist(password_lengths, bins=range(min(password_lengths), max(password_lengths) + 2), color="skyblue", edgecolor="black")
    plt.title(f"Histogram of Password Lengths for the First {LINE_READ_LIMIT} Passwords")
    plt.xlabel("Password Length")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0)
    plt.grid(axis="x", alpha=0)

    # Set xticks to each integer value in the range of password lengths
    plt.xticks(range(min(password_lengths), max(password_lengths) + 1))

    # Save the plot
    plt.savefig(f"{PLOTS_DIR}/password_length_histogram.png")
    logger.info(f"Saved histogram of password lengths to {PLOTS_DIR}/password_length_histogram.png")


def plot_word_length(spark_df) -> None:
    """
    Creates a box plot of the lengths of the words in the passwords.

    :param spark_df: Spark DataFrame containing a column of passwords.
    """
    logger.info("Creating box plot of word lengths")

    # Measure the length of each word
    spark_df = spark_df.withColumn("word_len", length(col("value")))

    # Show the length range
    min_len = spark_df.agg({"word_len": "min"}).collect()[0][0]
    max_len = spark_df.agg({"word_len": "max"}).collect()[0][0]
    avg_len = spark_df.agg({"word_len": "avg"}).collect()[0][0]
    median_len = spark_df.approxQuantile("word_len", [0.5], 0.01)[0]
    stddev_len = spark_df.agg({"word_len": "stddev"}).collect()[0][0]

    logger.info(f"Word length range: {min_len} to {max_len} characters")
    logger.info(f"Average word length: {avg_len:.2f} characters")
    logger.info(f"Median word length: {median_len:.2f} characters")
    logger.info(f"Standard deviation of word lengths: {stddev_len:.2f} characters")

    shortest_word = spark_df.filter(col("word_len") == min_len).first()["value"]
    longest_word = spark_df.filter(col("word_len") == max_len).first()["value"]
    logger.info(f"Shortest word: {shortest_word}")
    logger.info(f"Longest word: {longest_word}")

    # Remove passwords with word lengths greater than 16
    spark_df = spark_df.filter(col("word_len") <= 16)

    # Collect data for plotting
    word_lengths = spark_df.select("word_len").rdd.flatMap(lambda x: x).collect()

    # Plot the word length box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(word_lengths)
    plt.title(f"Box Plot of Word Lengths For the First {LINE_READ_LIMIT} Passwords (Words <= 16 Characters)")
    plt.ylabel("Password length")

    # Plot y-axis every 1 character
    plt.yticks(range(0, max(word_lengths) + 1, 1))

    # Set grid style
    plt.grid(axis="y", alpha=0.75)

    # Save the plot
    plt.savefig(f"{PLOTS_DIR}/word_length_boxplot.png")
    logger.info(f"Saved box plot of word lengths to {PLOTS_DIR}/word_length_boxplot.png")


def read_and_limit_data(filepath: str, limit: int=1000):
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
    # Read and limit data
    df = read_and_limit_data(ROCKYOU_FILE, LINE_READ_LIMIT)

    ###################
    ## Plot examples ##
    ###################

    # Plot password lengths histogram
    plot_password_lengths(df)

    # Plot character heatmap
    plot_character_heatmap(df, HEATMAP_CHARACTER_COUNT)

    # Plot word length box plot
    plot_word_length(df)

    # Find most common words
    find_most_common_words(df, 20)

    #################################
    ## Regular expression examples ##
    #################################

    # Fetch the count of numeric passwords
    df = fetch_passwords_using_regex(df, "^[0-9]+$")
    logger.info(f"Numeric passwords: {df.count()}")

    # Fetch alphabetic passwords
    df = fetch_passwords_using_regex(df, "^[A-Za-z]+$")
    logger.info(f"Alphabetic passwords: {df.count()}")

    # Fetch passwords with special characters
    df = fetch_passwords_using_regex(df, "[!@#$%^&*(),.?\":{}|<>]+")
    logger.info(f"Passwords with special characters: {df.count()}")

    # Fetch passwords that contain alphanumeric characters and special characters
    df = fetch_passwords_using_regex(df, "^(?=.*[A-Za-z0-9])(?=.*[!@#$%^&*(),.?\":{}|<>]).+$")
    logger.info(f"Passwords with alphanumeric and special characters: {df.count()}")

    # Fetch passwords that are shorter than 5 characters
    df = fetch_passwords_using_regex(df, "^.{1,5}$")
    logger.info(f"Passwords shorter than 5 characters: {df.count()}")

    # Fetch passwords that are between 6-8 characters
    df = fetch_passwords_using_regex(df, "^.{6,8}$")
    logger.info(f"Passwords between 6-8 characters: {df.count()}")

    # Fetch passwords with common patterns (e.g. "abc" or "123")
    df = fetch_passwords_using_regex(df, "(abc)")
    logger.info(f"Passwords with 'abc' patterns: {df.count()}")

    df = fetch_passwords_using_regex(df, "(123)")
    logger.info(f"Passwords with '123' patterns: {df.count()}")

    df = fetch_passwords_using_regex(df, "(012)")
    logger.info(f"Passwords with '012' patterns: {df.count()}")

    df = fetch_passwords_using_regex(df, "(qwe)")
    logger.info(f"Passwords with 'qwe' patterns: {df.count()}")

    df = fetch_passwords_using_regex(df, "(asd)")
    logger.info(f"Passwords with 'asd' patterns: {df.count()}")

    df = fetch_passwords_using_regex(df, "^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*(),.?\":{}|<>]).{8,}$")
    logger.info(f"Passwords with alphanumeric and special characters: {df.count()}")


if __name__ == "__main__":
    main()
