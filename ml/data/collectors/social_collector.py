"""
Social media data collector – Reddit (PRAW) and Twitter/X (Tweepy v2).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import praw
import tweepy
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class RedditCollector:
    """Stream and batch-fetch posts/comments from crypto subreddits."""

    SUBREDDITS = [
        "CryptoCurrency",
        "Bitcoin",
        "ethereum",
        "CryptoMarkets",
        "altcoin",
    ]

    def __init__(self) -> None:
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            user_agent=os.getenv("REDDIT_USER_AGENT", "CryptoPredictBot/1.0"),
        )
        logger.info("RedditCollector initialised (read-only).")

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30), reraise=True)
    def fetch_hot_posts(
        self,
        subreddit: str = "CryptoCurrency",
        limit: int = 100,
        symbol_filter: Optional[str] = None,
    ) -> list[dict]:
        """Return hot posts from a subreddit, optionally filtered by symbol keyword."""
        sub = self.reddit.subreddit(subreddit)
        posts = []
        for post in sub.hot(limit=limit):
            if symbol_filter and symbol_filter.lower() not in (
                post.title + post.selftext
            ).lower():
                continue
            posts.append(
                {
                    "id": post.id,
                    "title": post.title,
                    "text": post.selftext,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "created_utc": datetime.fromtimestamp(
                        post.created_utc, tz=timezone.utc
                    ).isoformat(),
                    "subreddit": subreddit,
                    "url": post.url,
                }
            )
        logger.info(f"Reddit r/{subreddit}: {len(posts)} posts.")
        return posts

    def fetch_all_subreddits(
        self, limit_per_sub: int = 50, symbol_filter: Optional[str] = None
    ) -> list[dict]:
        """Aggregate posts from all tracked subreddits."""
        all_posts: list[dict] = []
        for sub in self.SUBREDDITS:
            try:
                all_posts.extend(
                    self.fetch_hot_posts(
                        subreddit=sub,
                        limit=limit_per_sub,
                        symbol_filter=symbol_filter,
                    )
                )
            except Exception as exc:
                logger.warning(f"Reddit fetch failed for r/{sub}: {exc}")
        return all_posts


class TwitterCollector:
    """Fetch recent tweets about crypto via Twitter API v2."""

    CRYPTO_HASHTAGS: dict[str, str] = {
        "BTC": "#Bitcoin OR #BTC",
        "ETH": "#Ethereum OR #ETH",
        "SOL": "#Solana OR #SOL",
    }

    def __init__(self, bearer_token: Optional[str] = None) -> None:
        token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN", "")
        if not token:
            logger.warning("Twitter bearer token not configured – skipping.")
            self.client = None
        else:
            self.client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)
        logger.info("TwitterCollector initialised.")

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=60), reraise=True)
    def fetch_recent_tweets(
        self,
        symbol: str = "BTC",
        max_results: int = 100,
        exclude_retweets: bool = True,
    ) -> list[dict]:
        """Return recent tweets for the given symbol."""
        if not self.client:
            return []

        query_parts = [self.CRYPTO_HASHTAGS.get(symbol.upper(), f"#{symbol}")]
        if exclude_retweets:
            query_parts.append("-is:retweet")
        query_parts.append("lang:en")
        query = " ".join(query_parts)

        response = self.client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),
            tweet_fields=["created_at", "public_metrics", "lang"],
        )
        tweets = []
        if response.data:  # type: ignore[union-attr]
            for tw in response.data:  # type: ignore[union-attr]
                metrics = tw.public_metrics or {}
                tweets.append(
                    {
                        "id": str(tw.id),
                        "text": tw.text,
                        "created_at": tw.created_at.isoformat() if tw.created_at else "",
                        "retweet_count": metrics.get("retweet_count", 0),
                        "like_count": metrics.get("like_count", 0),
                        "reply_count": metrics.get("reply_count", 0),
                        "symbol": symbol.upper(),
                    }
                )
        logger.info(f"Twitter: {len(tweets)} tweets for {symbol}.")
        return tweets
