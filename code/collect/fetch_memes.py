"""Collect image posts from r/memes through Reddit's public JSON API (paper: Meme Collection).

Pages through the "top" listing with a weekly time filter (or the "hot" listing), keeps posts whose URL is an image
file, sorts them by score and writes the top N as CSV (title, image_url, score). The released pool (data/memes.csv)
was assembled from several such weekly snapshots.

  python fetch_memes.py --listing top --out memes_top_week.csv
  python fetch_memes.py --listing hot --out memes_hot.csv
"""
import argparse
import csv

import requests


def get_top_memes(listing, out, post_number, pages):
    all_posts = []
    after = None
    # Reddit returns at most 100 posts per request; page through the listing
    for page in range(pages):
        url = f"https://www.reddit.com/r/memes/{listing}.json"
        params = {"limit": 100, "t": "week"} if listing == "top" else {"limit": 100}
        if after:
            params["after"] = after

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        for post in data["data"]["children"]:
            post_data = post["data"]

            # only consider posts with image URLs
            url = post_data.get("url", "")
            if not any(url.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif"]):
                continue

            score = post_data["score"]  # score = upvotes - downvotes

            all_posts.append({
                "title": post_data["title"],
                "image_url": url,
                "score": score
            })

        after = data["data"].get("after")
        if not after:
            break

    # keep the top posts by score
    all_posts.sort(key=lambda x: x["score"], reverse=True)
    top = all_posts[:post_number]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "image_url", "score"])
        writer.writeheader()
        writer.writerows(top)

    print(f"saved {len(top)} memes to {out}")
    for i, post in enumerate(top[:5], 1):
        print(f"{i}. [{post['score']} score] {post['title'][:50]}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", choices=["top", "hot"], default="top")
    ap.add_argument("--out", default="memes_top_week.csv")
    ap.add_argument("--posts", type=int, default=500, help="keep this many posts (highest score first)")
    ap.add_argument("--pages", type=int, default=100, help="at most this many pages of 100 posts")
    a = ap.parse_args()
    get_top_memes(a.listing, a.out, a.posts, a.pages)
