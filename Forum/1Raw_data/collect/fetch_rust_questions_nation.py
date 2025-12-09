import pandas as pd
import psycopg2
import chardet

# Database connection configuration
DB_CONFIG = {
    "dbname": "rust_nation_data",
    "user": "postgres",
    "password": "github123",
    "host": "localhost",
    "port": "5432"
}

# Automatically detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)
    return chardet.detect(rawdata)['encoding']

# Create data tables
def create_tables(cur):
    cur.execute("DROP TABLE IF EXISTS question_stats;")
    cur.execute("DROP TABLE IF EXISTS forum_posts;")

    cur.execute("""
        CREATE TABLE question_stats (
            category TEXT,
            date DATE,
            question_count INTEGER
        );
    """)

    cur.execute("""
        CREATE TABLE forum_posts (
            id BIGINT PRIMARY KEY,
            "group" TEXT,
            slug TEXT,
            title TEXT,
            url TEXT,
            description TEXT,
            created_time TIMESTAMP
        );
    """)

# Import daily_stats.csv
def import_first_csv(cur, filename):
    encoding = detect_encoding(filename)
    print(f"[INFO] {filename} encoding detected as: {encoding}")

    df = pd.read_csv(
        filename,
        names=["category", "date", "question_count"],
        skiprows=1,  # Skip header
        encoding=encoding
    )

    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")

    for _, row in df.iterrows():
        values = []
        for v in row:
            if pd.isna(v):
                values.append(None)
            else:
                values.append(v)
        cur.execute(
            "INSERT INTO question_stats (category, date, question_count) VALUES (%s, %s, %s)",
            tuple(values)
        )

# Import topics.csv
def import_second_csv(cur, filename):
    encoding = detect_encoding(filename)
    print(f"[INFO] {filename} encoding detected as: {encoding}")

    # Read and remove all-empty columns
    df = pd.read_csv(filename, encoding=encoding, low_memory=False)
    df = df.dropna(axis=1, how='all')

    # Keep only the 7 needed columns
    needed_cols = ["id", "group", "slug", "title", "url", "description", "created_time"]
    df = df[needed_cols]

    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")

    for _, row in df.iterrows():
        values = []
        for v in row:
            if pd.isna(v):  # Convert NaN or NaT to None
                values.append(None)
            else:
                values.append(v)
        cur.execute("""
            INSERT INTO forum_posts (id, "group", slug, title, url, description, created_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, tuple(values))

# Main program
def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("Creating table structure...")
    create_tables(cur)

    print("Importing daily_stats.csv data...")
    import_first_csv(cur, "daily_stats.csv")

    print("Importing topics.csv data...")
    import_second_csv(cur, "topics.csv")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Data import completed!")

if __name__ == "__main__":
    main()
