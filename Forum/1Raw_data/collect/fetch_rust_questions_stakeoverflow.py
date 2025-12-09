# Scrape Rust questions from Stack Overflow and store them in the database
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import time

# Database connection configuration
DB_CONFIG = {
    "dbname": "rust_stake_data",
    "user": "postgres",
    "password": "github123",
    "host": "localhost",
    "port": "5432"
}

# Stack Exchange API Base URL
API_URL = "https://api.stackexchange.com/2.3/questions"
# Application required
API_KEY = ""  

# Create tables
def create_tables(conn):
    with conn.cursor() as cur:
        # Question data table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rust_questions (
                question_id     BIGINT PRIMARY KEY,
                title           TEXT,
                body            TEXT,
                tags            TEXT,
                creation_date   TIMESTAMP,
                link            TEXT,
                answer_count    INTEGER
            );
        """)
        # Fetch log table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fetch_log (
                fetch_date DATE PRIMARY KEY,
                success    BOOLEAN,
                count      INTEGER DEFAULT 0,
                error_message TEXT
            );
        """)
    conn.commit()

# Check if date has been fetched
def is_date_fetched(conn, date_str):
    with conn.cursor() as cur:
        cur.execute("SELECT success FROM fetch_log WHERE fetch_date = %s", (date_str,))
        row = cur.fetchone()
        return row and row[0] is True

# Log fetch result
def log_fetch_result(conn, date_str, success, count=0, error_message=None):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO fetch_log (fetch_date, success, count, error_message)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (fetch_date) DO UPDATE 
            SET success = EXCLUDED.success,
                count = EXCLUDED.count,
                error_message = EXCLUDED.error_message
        """, (date_str, success, count, str(error_message) if error_message else None))
    conn.commit()

# Save question data
def save_questions(conn, questions):
    if not questions:
        return
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO rust_questions (question_id, title, body, tags, creation_date, link, answer_count)
            VALUES %s
            ON CONFLICT (question_id) DO NOTHING
        """, questions)
    conn.commit()

# Fetch questions for a specific date
def fetch_questions_for_date(date_str):
    fromdate = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
    todate = fromdate + 86399  # End of the day
    has_more = True
    page = 1
    questions = []

    while has_more:
        params = {
            "page": page,
            "pagesize": 100,
            "fromdate": fromdate,
            "todate": todate,
            "order": "asc",
            "sort": "creation",
            "tagged": "rust",
            "site": "stackoverflow",
            "filter": "withbody"
        }
        if API_KEY:
            params["key"] = API_KEY

        r = requests.get(API_URL, params=params)
        if r.status_code == 429:
            print("⚠️ Rate limit triggered, pausing for 60 seconds...")
            time.sleep(60)
            continue
        if r.status_code != 200:
            raise Exception(f"API request failed: {r.status_code} {r.text}")

        data = r.json()
        for item in data.get("items", []):
            questions.append((
                item["question_id"],
                item["title"],
                item.get("body", ""),
                ",".join(item.get("tags", [])),
                datetime.utcfromtimestamp(item["creation_date"]),
                item.get("link", ""),
                item.get("answer_count", 0)
            ))

        has_more = data.get("has_more", False)
        page += 1
        time.sleep(0.5)  # Slow down requests to reduce ban probability

    return questions

# Main program
def main(start_date, end_date):
    conn = psycopg2.connect(**DB_CONFIG)
    create_tables(conn)

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    while current_date <= end_date_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        if is_date_fetched(conn, date_str):
            print(f"Skipping {date_str} (already fetched successfully)")
        else:
            try:
                print(f"Starting fetch for {date_str}...")
                questions = fetch_questions_for_date(date_str)
                save_questions(conn, questions)
                print(f"{date_str} fetch completed, total {len(questions)} items")
                log_fetch_result(conn, date_str, True, len(questions))
            except Exception as e:
                log_fetch_result(conn, date_str, False, 0, e)
                print(f"{date_str} fetch failed: {e}")
        current_date += timedelta(days=1)

    conn.close()

if __name__ == "__main__":
    # Set date range
    main(start_date="2022-05-15", end_date="2025-08-08")
