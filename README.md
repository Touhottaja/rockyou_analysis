# rockyou_analysis
A project to study big data processing and analysis using Python, Apache Spark, and rockyou.txt (as dataset).

## Goal
The goal of this study is to identify common password patterns via analyzing `rockyou.txt` and to craft adequate password policies.

## Setup
1. Create a new venv via `$ python3 -m venv [<your venv name>]`
2. Activate it via: `$ source [<your venv name>]/bin/activate`
3. Install requirements: `$ pip3 install -r requirements.txt`
4. Install java: `$ sudo apt install openjdk-11-jdk`
5. Download rockyou.txt (e.g., [from kaggle](https://www.kaggle.com/datasets/wjburns/common-password-list-rockyoutxt/data)) and place it in this directory.

## Running the analysis
Adjust the amount of read passwords by configuring the `LINE_READ_LIMIT` variable in `rockyou.py`. The `rockyou.txt` contains over 14 million passwords, so be mindful about how many passwords you want to read.

Check which analysis you want to run in the `main()` function of `rockyou.py`.

The analysis can be run: `$ python3 rockyou.py`.
