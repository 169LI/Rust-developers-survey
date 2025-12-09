import psycopg2
import json
import asyncio
import traceback
import sys
import re
from openai import AsyncOpenAI
from bs4 import BeautifulSoup

# -------------------
# Database Configuration
# -------------------
DB_CONFIG = {
    "dbname": "rust_nation_data",
    "user": "postgres",
    "password": "github123",
    "host": "localhost",
    "port": "5432",
    "connect_timeout": 10
}

# -------------------
# New Prompt Template (Ensure at least 1 tag)
# -------------------
# -------------------
# New Prompt Template (Ensure at least 1 tag)
# -------------------
PROMPT_TEMPLATE = """
Perform multi-label classification for Rust-related posts and generate **specific, descriptive tags**. 
Each post contains a title (title) and a preprocessed description (description) with plain text only. 
Use the following classification framework as guidance for generating tags:

1. Core Language Features: Involving core Rust concepts such as ownership, borrowing, lifetimes, generics, traits, macros, closures, enums, smart pointers, concurrency, async, error handling, control flow, pattern matching, module organization, foreign function interface, standard library, and collections.
2. Learning Resources and Community Support: Involving learning materials (books, videos, Chinese resources), documentation quality, community interaction (timeliness, centralization), and non-English community support.
3. Development Experience and Tools: Involving toolchain stability, compiler diagnostics, checker strictness, code refactoring, dependency management (Cargo), unsafe code, development tools (RustRover, VSCode), analysis/testing/debugging tools, compilation time, library quantity and quality, documentation completeness, and issues/maintenance of third-party libraries.
4. Use Cases and Cross-Platform Support: Involving Rust use cases, developer backgrounds, cross-platform deployment, containerization, virtualization, cross-platform debugging, and mobile/embedded support.
5. Performance and Language Comparison: Involving Rust’s performance advantages (memory safety, high performance), real-world application outcomes, comparisons with other languages (C++, Go, etc.), and differences in memory usage and safety.
6. Ecosystem and Future Development: Involving Rust ecosystem expansion, community support improvements, documentation enhancements, and future trends.

### Task:
- Analyze the given Rust-related post (title + description). Content may be in English or Chinese.
- Identify **all relevant topics** and generate descriptive tags.
- **Always output at least one tag.**
- If the content is vague or cannot be classified into specific fine-grained tags, 
  fall back to the **broad category tags** below instead of "general_rust":
  ["core_language_features", "learning_resources", "development_experience", 
   "application_scenarios", "performance_comparison", "ecosystem_future"]

- The output must strictly follow this format:

Tags: ["tag1", "tag2", ...]

Post Content:
Title: {{TITLE}}
Description: {{DESCRIPTION}}
"""


# -------------------
# Preprocess Description
# -------------------
def preprocess_description(description):
    if not description:
        return ""
    soup = BeautifulSoup(description, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[\*\#\>]+', '', text)
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    if len(text) > 200:
        text = text[:200] + "..."
    return text

# -------------------
# DeepSeek Call
# -------------------
client = AsyncOpenAI(
    api_key="",  # Replace with your key
    base_url="https://api.deepseek.com"
)

async def call_deepseek(prompt):
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=512,
                temperature=0.6,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                raise

# -------------------
# Label Extraction Function
# -------------------
def extract_labels(result_text):
    labels = []
    for line in result_text.splitlines():
        if line.strip().startswith("Tags:"):
            try:
                raw_labels = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
            except:
                raw_labels = [tag.strip() for tag in line.split(":", 1)[1].strip("[] ").split(",") if tag.strip()]
            for l in raw_labels:
                if l.strip():
                    labels.append(l.strip())
            break
    if not labels:
        labels = ["general_rust"]
    return labels

# -------------------
# Main Logic
# -------------------
async def process_retry_posts(batch_size=50, concurrent_limit=5):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        offset = 0
        total_fixed = 0

        sem = asyncio.Semaphore(concurrent_limit)

        async def process_single(qid, title, description):
            nonlocal total_fixed
            async with sem:
                processed_description = preprocess_description(description)
                prompt = PROMPT_TEMPLATE.replace("{{TITLE}}", title or "").replace("{{DESCRIPTION}}", processed_description or "")
                try:
                    result_text = await call_deepseek(prompt)
                    labels = extract_labels(result_text)
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE forum_posts SET generated_tags=%s WHERE id=%s",
                            (json.dumps(labels, ensure_ascii=False), qid)
                        )
                        conn.commit()
                    total_fixed += 1
                    print(f"[Fix Completed] id={qid}, Tags={labels}", flush=True)
                except Exception as e:
                    print(f"[Error] id={qid}, Error: {str(e)}", flush=True)
                    print(traceback.format_exc())

        while True:
            cursor.execute("""
                SELECT id, title, description
                FROM forum_posts
                WHERE (generated_tags = '[]')
                ORDER BY id
                LIMIT %s OFFSET %s
            """, (batch_size, offset))
            rows = cursor.fetchall()

            if not rows:
                break

            print(f"Start fixing {len(rows)} posts (offset={offset})", flush=True)

            tasks = [process_single(qid, title, description) for qid, title, description in rows]
            await asyncio.gather(*tasks)

            offset += batch_size

        cursor.close()
        conn.close()
        print(f"Retry script execution completed ✅ Total fixed {total_fixed} posts", flush=True)

    except Exception as e:
        print(f"[Main Program Error] {str(e)}", flush=True)
        print(traceback.format_exc())

# -------------------
# Run
# -------------------
if __name__ == "__main__":
    asyncio.run(process_retry_posts(batch_size=100, concurrent_limit=5))
