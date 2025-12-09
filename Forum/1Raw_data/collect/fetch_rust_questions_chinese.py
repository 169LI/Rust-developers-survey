import os
import time
import csv
# import aiohttp
import asyncio
import psycopg2
import urllib.parse
# from bs4 import BeautifulSoup
from datetime import datetime

# ------------------ Configurable Items ------------------
BASE_DOMAIN = "https://rustcc.cn"
BASE_SECTION_URL = "https://rustcc.cn/section"
CSV_FILE = "rustcc_posts.csv"
DB_NAME = "rust_cc_data"
DB_USER = "postgres"  # Modify as per your PostgreSQL setup
DB_PASSWORD = "github123"      # Modify as per your PostgreSQL setup
DB_HOST = "localhost"
DB_PORT = "5432"
PROGRESS_DIR = "progress"      # Checkpoint file directory
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds
REQUEST_TIMEOUT = 10
CONCURRENT_PAGES = 5  # Number of pages to fetch concurrently
CONCURRENT_POSTS = 10  # Number of post details to fetch concurrently

# Write Cookie here directly (paste the full string)
COOKIE_STRING = "" 

# Multi-section configuration
SECTIONS = [
    {"id": "498bfc50-3707-406f-b7ca-ede9cbf8808d", "group": "问答", "daily_stats": False},
    {"id": "751ee61a-a1d4-4bd1-87ca-4ca455d24a59", "group": "异步IO", "daily_stats": False},
    {"id": "c2511921-51f7-401f-a0c0-d3abcfa0631c", "group": "综合讨论", "daily_stats": False},
    {"id": "522c7491-6a5f-4141-a3a1-3070c0466586", "group": "Wasm", "daily_stats": False},
    {"id": "b5da3eae-a44c-44c2-ab34-bf49e290e257", "group": "Web开发框架", "daily_stats": False},
    {"id": "12987868-705f-4ce2-b158-4d43db7d3a97", "group": "区块链", "daily_stats": False},
    {"id": "f38f6ee2-9e28-455a-95a4-f959e9efa02d", "group": "机器学习", "daily_stats": False},
    {"id": "ad7f4769-63b6-4616-a44d-1a6fd60e0a2e", "group": "IoT", "daily_stats": False},
    {"id": "abef1881-1750-4da7-a2e7-71ab8f7e154b", "group": "Web和服务端开发", "daily_stats": False},
    {"id": "fde4792c-b6f2-4fb7-804b-74c022119d4f", "group": "微服务Service Mesh", "daily_stats": False},
    {"id": "3c8929e5-37f3-44e0-8d30-3c62898c0e50", "group": "Rust Web前端开发", "daily_stats": False},
]

# HEADERS (Hardcoded Cookie)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115 Safari/537.36",
    "Cookie": COOKIE_STRING
}

# ------------------ Database Related ------------------
def create_tables():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rust_questions (
            question_id TEXT PRIMARY KEY,
            title TEXT,
            tags TEXT,
            body TEXT,
            creation_date TEXT,
            link TEXT,
            group_name TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            fetch_date TEXT,
            group_name TEXT,
            success BOOLEAN,
            count INTEGER DEFAULT 0,
            error_message TEXT,
            PRIMARY KEY (fetch_date, group_name)
        );
    """)
    conn.commit()
    conn.close()

def save_post_to_db(question_id, title, tags, body, creation_date, link, group_name):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO rust_questions
        (question_id, title, tags, body, creation_date, link, group_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (question_id) DO NOTHING
    """, (question_id, title, tags, body, creation_date, link, group_name))
    conn.commit()
    conn.close()

def save_log_to_db(fetch_date, group_name, success, count, error_message=None):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO fetch_log (fetch_date, group_name, success, count, error_message)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (fetch_date, group_name) DO UPDATE
        SET success = EXCLUDED.success, count = EXCLUDED.count, error_message = EXCLUDED.error_message
    """, (fetch_date, group_name, success, count, error_message))
    conn.commit()
    conn.close()

def load_existing_links_from_db():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("SELECT link FROM rust_questions WHERE link IS NOT NULL")
        rows = cur.fetchall()
        conn.close()
        links = {r[0] for r in rows if r and r[0]}
        return links, len(links)
    except psycopg2.Error:
        return set(), 0

# ------------------ Network Requests ------------------
async def fetch_with_retry(session, url, params=None):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, params=params, timeout=REQUEST_TIMEOUT) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    print(f"[Error] Status code {resp.status}, retry {attempt}...")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"[Error] Request failed: {e}, retry {attempt}...")
        await asyncio.sleep(RETRY_DELAY)
    print(f"[Skip] Request {url} failed {MAX_RETRIES} times continuously, skipped.")
    return None

# ------------------ Parse Page ------------------
async def get_post_links(session, page, section_id, group_name):
    params = {"id": section_id, "current_page": page}
    text = await fetch_with_retry(session, BASE_SECTION_URL, params=params)
    if not text:
        return []
    soup = BeautifulSoup(text, "lxml")
    post_links = []
    for a in soup.find_all("a", class_="title left"):
        href = a.get("href")
        if not href:
            continue
        full_url = urllib.parse.urljoin(BASE_DOMAIN, href)
        parsed = urllib.parse.urlparse(href)
        q = urllib.parse.parse_qs(parsed.query)
        post_id = q.get("id", [None])[0] or parsed.path.split("/")[-1]
        title = a.get_text(strip=True)
        post_links.append((post_id, full_url, title, group_name))
    return post_links

async def get_post_detail(session, post_data):
    post_id, url, title, group_name = post_data
    text = await fetch_with_retry(session, url)
    if not text:
        return None
    soup = BeautifulSoup(text, "lxml")
    tags_elem = soup.find("small")
    tags = tags_elem.get_text(strip=True).replace("Tags：", "") if tags_elem else ""
    content_elem = soup.find("div", class_="detail-body")
    content = content_elem.get_text("\n", strip=True) if content_elem else ""
    time_elem = soup.find("span", class_="article_created_time")
    creation_date = time_elem.get_text(strip=True) if time_elem else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "question_id": post_id or f"{group_name}_{int(time.time()*1000)}",
        "title": title,
        "tags": tags,
        "body": content,
        "creation_date": creation_date,
        "link": url,
        "group_name": group_name
    }

# ------------------ CSV / Progress ------------------
def ensure_progress_dir():
    if not os.path.exists(PROGRESS_DIR):
        os.makedirs(PROGRESS_DIR)

def load_progress(group_name):
    ensure_progress_dir()
    filename = os.path.join(PROGRESS_DIR, f"progress_{group_name}.txt")
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except Exception:
            return 1
    return 1

def save_progress(group_name, page):
    ensure_progress_dir()
    filename = os.path.join(PROGRESS_DIR, f"progress_{group_name}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(page))

def save_to_csv(rows):
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["question_id", "title", "tags", "body", "creation_date", "link", "group_name"])
        writer.writerows(rows)

# ------------------ Main Process ------------------
async def process_section(sec, existing_links):
    sec_id = sec["id"]
    group_name = sec["group"]
    need_daily = bool(sec.get("daily_stats", False))
    print(f"\n===== Start scraping section: {group_name} (section id={sec_id}), daily_stats={need_daily} =====")
    start_page = load_progress(group_name)
    group_count = 0
    error_msg = None

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        try:
            page = start_page
            while True:
                # Fetch multiple pages concurrently
                page_range = range(page, page + CONCURRENT_PAGES)
                tasks = [get_post_links(session, p, sec_id, group_name) for p in page_range]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                all_posts = []
                has_data = False
                max_page = page

                for p, posts in zip(page_range, results):
                    if isinstance(posts, Exception):
                        print(f"[{group_name}] Page {p} request failed: {posts}")
                        continue
                    if not posts:
                        print(f"[{group_name}] No more posts on page {p}")
                        continue
                    has_data = True
                    all_posts.extend(posts)
                    max_page = max(max_page, p)

                if not has_data:
                    print(f"[{group_name}] No more pages, finishing this section.")
                    break

                # Fetch post details concurrently
                to_save_csv = []
                post_tasks = [get_post_detail(session, post_data) for post_data in all_posts if post_data[1] not in existing_links]
                post_results = await asyncio.gather(*post_tasks, return_exceptions=True)

                for post_data in post_results:
                    if isinstance(post_data, Exception):
                        print(f"[{group_name}] Detail page request failed: {post_data}")
                        continue
                    if post_data is None:
                        continue
                    save_post_to_db(
                        post_data["question_id"],
                        post_data["title"],
                        post_data["tags"],
                        post_data["body"],
                        post_data["creation_date"],
                        post_data["link"],
                        post_data["group_name"]
                    )
                    to_save_csv.append([
                        post_data["title"],
                        post_data["tags"],
                        post_data["body"],
                        post_data["group_name"]
                    ])
                    existing_links.add(post_data["link"])
                    group_count += 1

                if to_save_csv:
                    save_to_csv(to_save_csv)
                    print(f"[{group_name}] Saved {len(to_save_csv)} items to CSV/DB.")

                # Update progress to the highest successful page
                save_progress(group_name, max_page)
                page += CONCURRENT_PAGES
                await asyncio.sleep(1)  # Brief pause between batches to avoid overwhelming the server

        except Exception as e:
            error_msg = str(e)
            print(f"[{group_name}] Exception occurred: {e}")

    fetch_date = datetime.now().strftime("%Y-%m-%d")
    save_log_to_db(fetch_date, group_name, success=(error_msg is None),
                   count=group_count, error_message=error_msg)
    print(f"[{group_name}] Completed. Added {group_count} items; written to fetch_log.")
    return group_count

async def main():
    print("=== Start Crawler (Multi-section Mode) ===")
    create_tables()
    existing_links, initial_total = load_existing_links_from_db()
    print(f"Database has {initial_total} records (for deduplication).")
    total_new = 0

    for sec in SECTIONS:
        total_new += await process_section(sec, existing_links)

    print("\n=== All sections scraping completed ===")
    print(f"Total new records added this time: {total_new} (Database + CSV).")
    print("Finished.")

if __name__ == "__main__":
    asyncio.run(main())
